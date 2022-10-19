#!/bin/bash

# necessary to catch python errors
set -E

# Run on CPU (only if desired)
# export CUDA_VISIBLE_DEVICES="-1"

# Argument Parsing
##################

help()
{
  echo "Runs the cell segmentation and tracking pipeline"
  echo 
  echo "Syntax: run_pipeline.sh [options]"
  echo 
  echo "Options:"
  echo " -h, --help         Display this help"
  echo " --restart [PATH]   Restart pipeline from log file. If PATH is specified"
  echo "                    the checkpoint and settings file will be restored from"
  echo "                    PATH, otherwise the current working directory is searched"
  echo " --headless         Run pipeline in headless mode (no GUI)"
  echo " --loglevel         Set logging level of script (0-7), defaults to 7 (max log)"
  echo 
  exit 2
}


while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      help
      ;;
    --restart)
      RESTART="True"
      # check if we got path as values
      if [ -d "$2" ]; then
        RESTARTPATH="$2"
        shift # one extra shift for value
      # make sure the next one is an option, -v to check if it is set at all
      elif [ "$2" != "--*" ] && [ ! -z ${2+x} ]; then
        echo "Restart path does not exist: $2"
        exit 1
      fi
      shift # past argument
      ;;
    --headless)
      HEADLESS="True"
      shift # past argument
      ;;
    --loglevel)
      # check if we got an int by comparing the numer to its arithmetic eval
      if [[ $(( $2 )) != $2 ]] || [ $2 -lt "0" ]; then
        echo "--loglevel must be an int >= 0, got $2!"
        exit 1
      fi
      LOGLEVEL=$2
      shift # past argument
      shift # past value
      ;;
    -*|--*)
      echo "Unknown option $1"
      echo
      help
      exit 1
      ;;
    *)
      echo "No positional arguments accepted!"
      echo 
      help
      exit 1
      ;;
  esac
done

# Logging
#########

# reduce TF output 3 -> only errors, 0 -> print all
export TF_CPP_MIN_LOG_LEVEL="3"

# set verbose level
if [ ! -z ${LOGLEVEL+x} ]; then
  __VERBOSE=$LOGLEVEL
else
  __VERBOSE=7
fi

declare -a LOG_LEVELS
# https://en.wikipedia.org/wiki/Syslog#Severity_level
LOG_LEVELS=([0]="emerg" [1]="alert" [2]="crit" [3]="err" [4]="warning" [5]="notice" [6]="info" [7]="debug")
function .log () {
  local LEVEL=${1}
  shift
  if [ ${__VERBOSE} -ge ${LEVEL} ]; then
    echo "[${LOG_LEVELS[$LEVEL]}]" "$@"
  fi
}

# Mac M1 Handling
#################

# Check if we are on a Mac and if there is an M1
if [[ ${OSTYPE} = darwin* ]] && [[ $(uname -m) == "arm64" ]]; then
  # Check if there is the right miniforge installation
  if [[ -d $HOME'/miniforge3/envs/workflow/bin' ]]; then
    PYTHON_EXE=$HOME'/miniforge3/envs/workflow/bin/python3'
  else
    # Exit
    .log 2 "You are using an M1 Mac but there is no miniforge installtion in: $HOME/miniforge3/envs/workflow/bin!"
    .log 2 "Please install the correct environment. Exiting..."
    exit 1
  fi
else
  PYTHON_EXE=python
fi

# Checkpointing
###############

# We clear the log if we are successful
trap 'clear_log' EXIT
# If there is an error or interrupt we log
trap 'log_checkpoint $current_func' ERR SIGINT SIGHUP SIGKILL SIGTERM

# Directory of the checkpoint for copy and file name
CHECKDIR="./"
CHECKLOG="checkpoints.log"

clear_log() {
  # make sure the last exit code was 0
  if [ $? -eq 0 ] ; then
    rm -f "$CHECKLOG"
  fi
}

log_checkpoint() {
  # Print fail and log
  if [ "$1" == "" ]; then
    .log 2 "Unexpected error outside of callstack!"
  else
    .log 2 "Error while running: $1"
  fi
  echo "$1" > "$CHECKLOG"
  # copy checkpoint and setting to current path
  # use are sync to avoid possible "are the same file" 
  .log 2 "Copy checkpoint and settings to: ${CHECKDIR}"
  rsync "${CHECKLOG}" "${CHECKDIR}/${CHECKLOG}"
  rsync settings.sh "${CHECKDIR}/settings.sh"
  exit 1 
}

retry() {
  # transfer the function name into the current current_func
  current_func=$1
  # If there is no checkpoint file, there is nothing to do
  [ ! -f "$CHECKLOG" ] && return 0
  # if we have a checkpoint file we return 0 if it contains the current function
  if grep -q "$1" "$CHECKLOG"; then
    .log 6 "retry $1"; rm "$CHECKLOG"; return 0
  else
    .log 6 "skip $1"; return 1
  fi
}

# Pipeline
##########

set_parameters() {
  # Start GUI only if we are not in headless mode and no retry
  if [ "$HEADLESS" != "True" ] && retry ${FUNCNAME[0]}; then
    .log 7 "Starting up the GUI"
    ${PYTHON_EXE} apps/set_parameters.py
  fi
  # In case of headless or checkpoint we just source the settings
  if [ -f "settings.sh" ]; then
    .log 7 "Sourcing parameters 'settings.sh'"
    source settings.sh
  else
    .log 3 "Running in headless mode, but no settings.sh found!"
    exit 1
  fi
}

restrict_frames_family() {
  # Starts up the script to restrict the frames (arg is pos)
  retry "${FUNCNAME[0]}_$1" || return 0
  .log 6 "Restricting frames for identifier: ${POS}"
  ${PYTHON_EXE} apps/restrict_frames.py
  source settings.sh
}

source_paths_family() {
  # specify different folders needed for segmentation and tracking
  RAW_IM="raw_im/"
  CUT_PATH="cut_im/"
  SEG_IM_PATH="seg_im/"
  SEG_IM_TRACK_PATH="input_ilastik_tracking/"
  TRACK_OUT_PATH="track_output/"
}

setup_folders_family() {
  # creates the folder structure for the family machine

  # Set path for checkpoints 
  CHECKDIR="$PATH_FOLDER$POS"
   
  # only redo this if necessary (arg is POS again)
  retry "${FUNCNAME[0]}_$1" || return 0
  
  .log 6 "Generating folder structure..."

  # Delete results folder for this position in case it already exists.
  # In this way the segmentation can be rerun
  rm -rf "$PATH_FOLDER$POS"

  # generate folders for different channels (phase, fluorescent)
  mkdir -p "$PATH_FOLDER$POS"

  for i in $(seq 1 $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    # Base folder for different channels
    mkdir -p "$PATH_FOLDER$POS/${!CH}/"
    # raw images
    mkdir -p "$PATH_FOLDER$POS/${!CH}/$RAW_IM"
    # cutouts
    mkdir -p "$PATH_FOLDER$POS/${!CH}/$CUT_PATH"
    # segmentation images
    mkdir -p "$PATH_FOLDER$POS/${!CH}/$SEG_IM_PATH"
    # stack of segmentation images for tracking
    mkdir -p "$PATH_FOLDER$POS/${!CH}/$SEG_IM_TRACK_PATH"
    # tracking output (Unet)
    mkdir -p "$PATH_FOLDER$POS/${!CH}/$TRACK_OUT_PATH"
  done
}

copy_files_family() {
  # Copies the necessary files into the folder structure

  # Checkpoint, arg is the pos
  retry "${FUNCNAME[0]}_$1" || return 0

  # Channel loop
  .log 6 "Copying files for identifier: ${POS}"
  for i in $(seq 1 $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    VAR=$(find "$PATH_FOLDER" -maxdepth 1 -name *"$POS"*"${!CH}"*".$FILE_TYPE")
    # Catch the copy output for debug logging
    # We need to split it in two otherwise the error wont be catched correctly!
    local COPYLOG
    COPYLOG=$(cp -v "$VAR" "$PATH_FOLDER$POS/${!CH}/")
    .log 7 "$COPYLOG"
  done
}

split_frames_family() {
  # Splits the frames for the family machine per identifier

  # Checkpoint, arg is the pos
  retry "${FUNCNAME[0]}_$1" || return 0

  # Channel loop
  .log 6 "Splitting frames for identifier: ${POS}"
  for i in $(seq 1 $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    INP=$(find "$PATH_FOLDER$POS/${!CH}/" -name *".$FILE_TYPE")
    ${PYTHON_EXE} apps/stack2frames.py --path "$INP" --start_frame "$START_FRAME" --end_frame "$END_FRAME" --deconv "$DECONVOLUTION" --loglevel "${__VERBOSE}"
  done
}

cut_chambers_family() {
  # Cuts the chambers for the family machine per identifier

  # Checkpoint, arg is the pos
  retry "${FUNCNAME[0]}_$1" || return 0

  # Split for number of channels
  .log 6 "Cutting chambers for identifier: ${POS}"
  
  # Collect the arguments
  local ARGS=""
  for i in $(seq 1 $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    ARGS="${ARGS} \"${PATH_FOLDER}${POS}/${!CH}/${RAW_IM}\""
  done
  
  # We need to do the eval turn to deal with paths that have spaces!
  CMD="${PYTHON_EXE} apps/frames2cuts.py --cutout_class $CHAMBER_CUTOUT --channel $ARGS"
  eval $CMD
}

segmentation_family() {
  # Performs the image segmentation for the family machine

  # Checkpoint, arg is the pos
  retry "${FUNCNAME[0]}_$1" || return 0

  # Phase segmention dependent channel loops
  .log 6 "Segmenting images for identifier: ${POS}"
  # Set the start of the loop
  local START=2
  if [ "$PHASE_SEGMENTATION" == "True" ]; then
    local START=1
  fi
  # cycle through channels
  for i in $(seq $START $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    ${PYTHON_EXE} apps/main_prediction.py --path_model_weights '../model_weights/model_weights_family_mother_machine/' --path_pos "$PATH_FOLDER$POS" --path_channel "${!CH}" --segmentation_class "${SEGMENTATION_METHOD}" --postprocessing
    ${PYTHON_EXE} apps/analyse_segmentation.py --path_seg "$PATH_FOLDER$POS/${!CH}/$SEG_IM_PATH/" --path_result "$PATH_FOLDER$POS/${!CH}/"  --loglevel "${__VERBOSE}"
  done
}

tracking_family() {
  # Cell tracking for the family machine

  # Checkpoint, arg is the pos
  retry "${FUNCNAME[0]}_$1" || return 0

  # Phase segmentation dependent channel loops
  .log 6 "Running cell tracking for identifier: ${POS}"
  # Set the start of the loop
  local START=2
  if [ "$PHASE_SEGMENTATION" == "True" ]; then
    local START=1
  fi
  # cycle through channels
  for i in $(seq $START $NUM_CHANNEL_TYPES); do
    CH="CHANNEL_$i"
    ${PYTHON_EXE} apps/track_cells_crop.py --path "$PATH_FOLDER$POS/${!CH}/" --loglevel "${__VERBOSE}"
    ${PYTHON_EXE} apps/generate_lineages.py --path "$PATH_FOLDER$POS/${!CH}/$TRACK_OUT_PATH" --loglevel "${__VERBOSE}"
  done

  # we copy the current settings.sh into the path folder for reproducibility
  COPYLOG=$(cp -v settings.sh "${PATH_FOLDER}/${POS}/settings.sh")
  .log 7 "$COPYLOG"   
}

source_paths_well() {
  # Sets the paths for the WELL run

  # Some bash string manipulations
  PATH_FILE_WO_EXT="${PATH_FILE%.*}"
  FILE_NAME="${PATH_FILE##*/}"

  .log 6 "Extracted path without ext: $PATH_FILE_WO_EXT"
  .log 6 "Extracted filename: $FILE_NAME"

  # Set directories
  RAW_IM="raw_im/"
  SEG_PATH="xy1/"
  CUT_PATH="phase/"
  SEG_IM_PATH="seg_im/"
  SEG_MAT_PATH="seg/"
  SEG_IM_TRACK_PATH="input_ilastik_tracking/"
}

setup_folders_well() {
  # creates the folder structure for the well machine
  
  # checkpoint, only redo if necessary
  retry "${FUNCNAME[0]}" || return 0
  
  .log 6 "Generating folder structure..."
  # delete results folder in case it already exists
  rm -rf "$PATH_FILE_WO_EXT"

  # generate folder to store the results
  mkdir -p "$PATH_FILE_WO_EXT"
  cp "$PATH_FILE" "$PATH_FILE_WO_EXT"

  # generate folders raw_im
  mkdir -p "$PATH_FILE_WO_EXT/$RAW_IM"
  # generate folders for tracking results
  mkdir -p "$PATH_FILE_WO_EXT/$SEG_PATH"
  # generate folders for cutout images
  mkdir -p "$PATH_FILE_WO_EXT/$SEG_PATH$CUT_PATH"
  # generate folders for segmentation images
  mkdir -p "$PATH_FILE_WO_EXT/$SEG_IM_PATH"
  # generate folder seg_im_track for stacks of segmentation images for tracking
  mkdir -p "$PATH_FILE_WO_EXT/$SEG_IM_TRACK_PATH"
  # generate folders for segmentation-mat files
  mkdir -p "$PATH_FILE_WO_EXT/$SEG_PATH$SEG_MAT_PATH"
}

split_frames_well() {
  # Splits the frames for the WELL

  # Checkpoint
  retry "${FUNCNAME[0]}" || return 0

  .log 6 "Spliting frames..."
  ${PYTHON_EXE} stack2frames.py --path "$PATH_FILE_WO_EXT/$FILE_NAME" --pos "" --channel "" --start_frame "$START_FRAME" --end_frame "$END_FRAME" --deconv "$DECONVOLUTION"
  # Catch the copy output for debug logging
  # We need to split it in two otherwise the error wont be catched correctly!
  local COPYLOG
  COPYLOG=$(cp -v "$PATH_FILE_WO_EXT/$RAW_IM"*".$FILE_TYPE" "$PATH_FILE_WO_EXT/$SEG_PATH$CUT_PATH")
  .log 7 "${COPYLOG}"
}

segmentation_well() {
  # Performs the image segmentation for the WELL

  # Checkpoint
  retry "${FUNCNAME[0]}" || return 0

  .log 6 "Segmenting images..."
  ${PYTHON_EXE} main_prediction.py --path_model_weights '../model_weights/model_weights_well/' --path_pos "$PATH_FILE_WO_EXT" --path_channel "" --postprocessing 1
}

conversion_well() {
  # Performs file conversion for the WELL

  # Checkpoint
  retry "${FUNCNAME[0]}" || return 0

  .log 6 "Run file-conversion..."
  ${PYTHON_EXE} seg2mat.py --path_cut "$PATH_FILE_WO_EXT/$SEG_PATH$CUT_PATH" --path_seg "$PATH_FILE_WO_EXT/$SEG_IM_PATH" --path_channel "$PATH_FILE_WO_EXT/"
}

tracking_well() {
  # Cell tracking for the WELL

  # Checkpoint
  retry "${FUNCNAME[0]}" || return 0

  .log 6 "Running the tracking..."
  # delete all files related to SuperSegger to ensure that SuperSegger runs
  rm -f "$PATH_FILE_WO_EXT/CONST.mat"
  rm -f "$PATH_FILE_WO_EXT/$SEG_PATH/clist.mat"
  rm -f "$PATH_FILE_WO_EXT/$SEG_PATH$SEG_MAT_PATH/"*"_err.mat"
  rm -fr "$PATH_FILE_WO_EXT/$SEG_PATH/cell"
  rm -f "$PATH_FILE_WO_EXT/$SEG_PATH/$RAW_IM/cropbox.mat"

  # Run matlab
  $MATLAB_ROOT/bin/matlab -nodisplay -r "tracking_supersegger('$PATH_FILE_WO_EXT', '$CONSTANTS' , $NEIGHBOR_FLAG, $TIME_STEP, $MIN_CELL_AGE, '$DATA_TYPE')"

  MAT_FILE="$PATH_FILE_WO_EXT/$SEG_PATH/clist.mat"
  # as long as 'clist.mat' is missing (hint for failed SuperSegger) the tracking can be repeated with a reduced number of frames
  while ! test -f "$MAT_FILE"; do
    rm -f "$PATH_FILE_WO_EXT/$SEG_PATH$SEG_MAT_PATH/"*"_err.mat"
    rm -f "$PATH_FILE_WO_EXT/CONST.mat"
    rm -f "$PATH_FILE_WO_EXT/$SEG_PATH/$RAW_IM/cropbox.mat"

    ${PYTHON_EXE} restrict_frames.py
    source settings.sh
    LIST_FILES=($(ls "$PATH_FILE_WO_EXT/$SEG_PATH$SEG_MAT_PATH"))
    NUM_FILES=${#LIST_FILES[@]}
    NUM_REMOVE=$NUM_FILES-$END_FRAME #number of files to remove

    for FILE in ${LIST_FILES[@]:$END_FRAME:$NUM_REMOVE}; do
      rm -f "$PATH_FILE_WO_EXT/$SEG_PATH$SEG_MAT_PATH/$FILE"
    done
    $MATLAB_ROOT/bin/matlab -nodisplay -r "tracking_supersegger('$PATH_FILE_WO_EXT', '$CONSTANTS' , $NEIGHBOR_FLAG, $TIME_STEP, $MIN_CELL_AGE, '$DATA_TYPE')"
  # END WHILE
  done

}


# Callstack
###########

# If the restart option is not set, we delete the log file
if [ "$RESTART" != "True" ]; then
  .log 7 "Clearing checkpoing file"
  clear_log
elif [ ! -z ${RESTARTPATH+x} ]; then
  # get the checkout 
  RESTORE_CHECKPOINT=$(find "$RESTARTPATH" -type f -name "$CHECKLOG")
  # get the settings.sh
  if [ -f "$RESTORE_CHECKPOINT" ]; then
    RESTORE_SETTINGS=$(dirname "${RESTORE_CHECKPOINT}")/settings.sh
  fi
  # check if single file exists
  if [ -f "$RESTORE_CHECKPOINT" ] && [ -f "$RESTORE_SETTINGS" ]; then
    .log 6 "Found checkpoint:  $RESTORE_CHECKPOINT"
    .log 6 "Found settings.sh: $RESTORE_SETTINGS"
    # restore the files, rsync in case somebode sets path to ./
    rsync "$RESTORE_CHECKPOINT" "$CHECKLOG"
    rsync "$RESTORE_SETTINGS" settings.sh
    source settings.sh
  else
    .log 6 "No checkpoint or settings.sh file found, starting again..."
    # clear log 
    clear_log
  fi
# END RESTARTPATH set
fi

# set the parameters
set_parameters

# Family Machine case
if [[ $DATA_TYPE == "FAMILY_MACHINE" ]]; then
  # check if path folder actually exists
  if [ -d "$PATH_FOLDER" ]; then
    .log 7 "Working in: $PATH_FOLDER"
  else
    .log 3 "Path folder does not exist: $PATH_FOLDER"
    exit 1
  fi

  # extract different positions from one dataset
  POSITIONS=()
  for i in "$PATH_FOLDER"*".$FILE_TYPE"; do
    # Command grouping {} prevents grep to throw an error if nothing was found
    POS=$(echo $i | { grep -Eo "${POS_IDENTIFIER}[0-9]+" || true; })
    POSITIONS+=($POS)
  done
  # Keep only unique 
  POS_UNIQ=($(printf "%s\n" "${POSITIONS[@]}" | sort -u));
  
  # See if we got anything
  if [ ${#POS_UNIQ[@]} -eq 0 ]; then
    .log 3 "Could not extract any file matching identifier: '${POS_IDENTIFIER}'"
    clear_log
    exit 1
  fi
  .log 7 "Extraced Identifiers: ${POS_UNIQ[@]}"

  # source the path names for all the folders
  source_paths_family

  # cycle through all identifiers
  for POS in "${POS_UNIQ[@]}"; do
    .log 7 "Starting with: ${POS}"

   if [[ $RUN_OPTION == "BOTH" ]] || [[ $RUN_OPTION == "SEGMENTATION" ]]; then
      # 1) Generate folder structure
      setup_folders_family $POS

      # 1.1) restrict frames for each position separately
      if  [ $POS != "${POS_UNIQ[0]}" ]; then
        restrict_frames_family $POS
      fi
    
       # 2) Copy files
      copy_files_family $POS

      # 3) Split frames
      split_frames_family $POS
    
      # 4) Cut chambers
      cut_chambers_family $POS

      # 5) Segmentation
      segmentation_family $POS
    fi

    # 6) Tracking
    if [[ $RUN_OPTION == "BOTH" ]] || [[ $RUN_OPTION == "TRACKING" ]]; then
      tracking_family $POS
    fi

  # End POS_UNIQ Loop
  done
# END FAMILY_MACHINE
fi

# TODO: Copy logs for WELL etc.

# Well Case
if [[ $DATA_TYPE == "WELL" ]]; then
  source_paths_well  
  
  if [[ $RUN_OPTION == "BOTH" ]] || [[ $RUN_OPTION == "SEGMENTATION" ]]; then
    # 1) Generate folder structure
    setup_folders_well

    # 2) Split frames
    split_frames_well

    # 3) Segmentation
    segmentation_well

    # 4) Conversion
    conversion_well
  fi

  # 5) Tracking
  if [[ $RUN_OPTION == "BOTH" ]] || [[ $RUN_OPTION == "TRACKING" ]]; then
    tracking_well
  fi 

# EMD WELL
fi

exit 0
import sys

import matplotlib
import tensorflow as tf

# backend has to be set before the pyplot import, TkAgg is compatible with most clusters
matplotlib.use("TkAgg")

# TF >= 2.16 ships Keras 3 as tf.keras, which breaks the Keras 2 API used
# throughout MIDAP's network code (functional model construction, add_loss with
# dangling inputs, loading Keras 2 .hdf5 weights, etc.).
# tf-keras is the official Keras 2 compatibility package; redirect every
# tensorflow.keras import path to it so no network file needs to change.
try:
    import tf_keras
    import tf_keras.backend
    import tf_keras.callbacks
    import tf_keras.layers
    import tf_keras.losses
    import tf_keras.metrics
    import tf_keras.models
    import tf_keras.optimizers

    tf.keras = tf_keras
    sys.modules["tensorflow.keras"] = tf_keras
    sys.modules["tensorflow.keras.backend"] = tf_keras.backend
    sys.modules["tensorflow.keras.callbacks"] = tf_keras.callbacks
    sys.modules["tensorflow.keras.layers"] = tf_keras.layers
    sys.modules["tensorflow.keras.losses"] = tf_keras.losses
    sys.modules["tensorflow.keras.metrics"] = tf_keras.metrics
    sys.modules["tensorflow.keras.models"] = tf_keras.models
    sys.modules["tensorflow.keras.optimizers"] = tf_keras.optimizers
except ImportError:
    # TF < 2.16: tf.keras is already Keras 2, nothing to do.
    pass

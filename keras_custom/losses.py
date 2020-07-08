import tensorflow as tf

"""
Custom losses
"""

def reconstruction(y_true, y_pred):
    return tf.keras.losses.MSE(y_true, y_pred)
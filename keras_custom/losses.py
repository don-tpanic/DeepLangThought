import tensorflow as tf

"""
Custom losses
"""

def total_loss(y_true, y_pred):

    # semantic_true = y_true[0]
    # semantic_pred = y_pred[0]
    # semantic_loss = tf.keras.losses.MSE(semantic_true, semantic_pred)



    # discrete_true = y_true[1]
    # discrete_pred = y_pred[1]
    discrete_loss = tf.keras.losses.CategoricalCrossentropy()(y_true, y_pred)
    
    return discrete_loss


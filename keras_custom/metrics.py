import tensorflow as tf


def totalLoss(y_true, y_pred):

    semantic_true = y_true[0]
    semantic_pred = y_pred[0]
    semantic_loss = tf.keras.metrics.MSE(semantic_true, semantic_pred)



    discrete_true = y_true[1]
    discrete_pred = y_pred[1]    
    discrete_loss = tf.keras.metrics.categorical_crossentropy(discrete_true, discrete_pred)

    print('from inside totalLoss')
    return semantic_loss + discrete_loss
from tensorflow.keras import losses
from tensorflow.keras import backend as K

"""
Custom losses
"""

def pearson_loss(y_true, y_pred):
    cov = K.mean((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    rho = cov / (K.std(y_true) * K.std(y_pred))
    return 1 - rho
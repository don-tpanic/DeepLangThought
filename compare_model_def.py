import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras_custom.models.language_model import lang_model, lang_model_contrastive
from TRAIN.utils.data_utils import load_config

# lang_model(w2_depth=2)



config = load_config(config_version='vgg16_finegrain_v1.1.run1')
lang_model_contrastive(config)
import os

# path names
_dirname = os.path.dirname(__file__)
ROOT_PATH = os.path.join(_dirname, '../..')
DATA_PATH = os.path.join(ROOT_PATH, './databroken/')
SUBMISSIONS_PATH = os.path.join(ROOT_PATH, './submissions/')
MODELS_PATH = os.path.join(ROOT_PATH, './models/')

# import configurations
CSV_SEP = ';'
CSV_ENCODING = 'latin-1'

# load model weights?
_LOAD_MODEL = True
_MODEL_NAME = '0.55RMSE_20190611-1736_model.h5'

def getModelPath():
    if _LOAD_MODEL:
        return os.path.join(MODELS_PATH, _MODEL_NAME)
    else:
        return ''
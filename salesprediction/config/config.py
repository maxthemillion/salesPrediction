import os

# path names
_dirname = os.path.dirname(__file__)
ROOT_PATH = os.path.join(_dirname, '../..')
DATA_PATH = os.path.join(ROOT_PATH, './data/')
EXPORT_PATH = os.path.join(ROOT_PATH, './exports/')

# import configurations
CSV_SEP = ';'
CSV_ENCODING = 'latin-1'
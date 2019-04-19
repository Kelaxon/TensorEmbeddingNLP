import os

def GetConfig(option):
    global  CONFIG
    CONFIG = {}
    if option=='16000oneliners':
        CONFIG['ROOT_PATH'] = './data/16000 oneliners/'
        CONFIG['RAW_DATA'] = './data/16000 oneliners/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/16000 oneliners/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/16000 oneliners/tensor_embedding.pkl'
        CONFIG['LOAD_DATA_METHOD'] = '_Load16000Oneliners'
        CONFIG['TOKEN_MODE'] = 'DEFAULT'

    if option=='semeval2007':
        CONFIG['ROOT_PATH'] = './data/AffectiveText.Semeval.2007/'
        CONFIG['RAW_DATA'] = './data/AffectiveText.Semeval.2007/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/AffectiveText.Semeval.2007/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/AffectiveText.Semeval.2007/tensor_embedding.pkl'
        CONFIG['TOKEN_MODE'] = 'DEFAULT'

    if option == 'authorProfiling':
        CONFIG['ROOT_PATH'] = './data/author profiling/pan15-author-profiling-training-dataset-2015-04-23-2' \
                              '/pan15-author-profiling-training-dataset-english-2015-04-23'
        CONFIG['RAW_DATA'] = os.path.join(CONFIG['ROOT_PATH'], 'data/raw.pkl')
        CONFIG['SAVED_RAW_DATA'] = os.path.join(CONFIG['ROOT_PATH'], 'pre_load.pkl')
        CONFIG['TENSOR_EMBEDDING'] = os.path.join(CONFIG['ROOT_PATH'], 'tensor_embedding.pkl')
        CONFIG['LOAD_DATA_METHOD'] = '_LoadAuthorProfiling'
        CONFIG['TOKEN_MODE'] = 'URL'
    return CONFIG



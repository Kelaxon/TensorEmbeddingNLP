def GetConfig(option):
    CONFIG = {}
    if option=='16000oneliners':
        CONFIG['ROOT_PATH'] = './data/16000 oneliners/'
        CONFIG['RAW_DATA'] = './data/16000 oneliners/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/16000 oneliners/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/16000 oneliners/tensor_embedding.pkl'
    return CONFIG
def GetConfig(option):
    CONFIG = {}
    if option=='16000oneliners':
        CONFIG['ROOT_PATH'] = './data/16000 oneliners/'
        CONFIG['RAW_DATA'] = './data/16000 oneliners/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/16000 oneliners/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/16000 oneliners/tensor_embedding.pkl'

    if option=='semeval2007':
        CONFIG['ROOT_PATH'] = './data/AffectiveText.Semeval.2007/'
        CONFIG['RAW_DATA'] = './data/AffectiveText.Semeval.2007/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/AffectiveText.Semeval.2007/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/AffectiveText.Semeval.2007/tensor_embedding.pkl'
        CONFIG['RESULT'] = './data/AffectiveText.Semeval.2007/result.mat'
    return CONFIG
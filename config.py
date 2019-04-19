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

    if option=='SST1':
        CONFIG['ROOT_PATH'] = './data/SST1/'
        CONFIG['RAW_DATA'] = './data/SST1/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/SST1/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/SST1/tensor_embedding.pkl'

    if option=='SST2':
        CONFIG['ROOT_PATH'] = './data/SST2/'
        CONFIG['RAW_DATA'] = './data/SST2/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/SST2/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/SST2/tensor_embedding.pkl'

    if option=='CR':
        CONFIG['ROOT_PATH'] = './data/CR/'
        CONFIG['RAW_DATA'] = './data/CR/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/CR/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/CR/tensor_embedding.pkl'

    if option=='SUBJ':
        CONFIG['ROOT_PATH'] = './data/SUBJ/'
        CONFIG['RAW_DATA'] = './data/SUBJ/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/SUBJ/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/SUBJ/tensor_embedding.pkl'

    if option=='Pun':
        CONFIG['ROOT_PATH'] = './data/Pun/'
        CONFIG['RAW_DATA'] = './data/Pun/data/raw.pkl'
        CONFIG['SAVED_RAW_DATA'] = './data/Pun/pre_load.pkl'
        CONFIG['TENSOR_EMBEDDING'] = './data/Pun/tensor_embedding.pkl'
    return CONFIG
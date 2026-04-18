'''
    配置文件
'''

# 目录名
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent # 项目根目录
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
LOGS_DIR = ROOT_DIR / 'logs'
MODELS_DIR = ROOT_DIR / 'models'

# 文件名
RAW_DATA_FILE = 'cmn.txt'
TRAIN_DATA_FILE = 'train.jsonl'
TEST_DATA_FILE = 'test.jsonl'
ZH_VOCAB_FILE = 'zh_vocab.txt'
EN_VOCAB_FILE = 'en_vocab.txt'
BEST_MODEL = 'best_model.pt'

# 特殊token
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'

# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 128

# 模型结构超参数
DIM_MODEL = 128
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
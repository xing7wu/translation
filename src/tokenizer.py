'''
    分词器
'''
from nltk import TreebankWordTokenizer, TreebankWordDetokenizer
from config import *

# 基分词器
class BaseTokenizer:
    # 初始化
    def __init__(self, vocab_list):
        self.vocab_size = len(vocab_list)
        # 保存词表（双向映射）
        self.id2word = vocab_list
        self.word2id = { word:id for id, word in enumerate(self.id2word) }
        # 记录特殊token的id
        self.unk_id = self.word2id[UNK_TOKEN]
        self.pad_id = self.word2id[PAD_TOKEN]
        self.sos_id = self.word2id[SOS_TOKEN]
        self.eos_id = self.word2id[EOS_TOKEN]

    # 定义工厂方法
    @classmethod
    def from_vocab(cls, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [ line.strip() for line in f.readlines()]
        return cls(vocab_list)

    # 创建词表：传入语料，词表文件路径
    @classmethod
    def build_vocab(cls, sentences, vocab_file):
        # 处理语料，分词，保存到集合
        vocab_set = set()
        for sentence in sentences:
            tokens = cls.tokenize(sentence)
            vocab_set.update(tokens)
        # 转换成词的列表，增加特殊token
        vocab_list = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN] + list(vocab_set)
        print("Vocab size: ", len(vocab_list))
        # 保存词表文件
        with open(vocab_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vocab_list))

    # 分词
    @classmethod
    def tokenize(cls, text) -> list[str]:
        pass

    # 编码（id化）
    def encode(self, text, mark=False):
        tokens = self.tokenize(text)
        # 如果是目标序列，直接添加sos和eos
        if mark:
            tokens = [SOS_TOKEN] + tokens + [EOS_TOKEN]
        ids = [ self.word2id.get(token, self.unk_id) for token in tokens ]
        return ids

# 中文分词器
class ZhTokenizer(BaseTokenizer):
    # 字符级分词
    @classmethod
    def tokenize(cls, text) -> list[str]:
        return list(text)

# 英文分词器
class EnTokenizer(BaseTokenizer):
    tokenizer = TreebankWordTokenizer()
    detokenizer = TreebankWordDetokenizer()

    @classmethod
    def tokenize(cls, text) -> list[str]:
        return cls.tokenizer.tokenize(text)

    # 解码，传入id列表，合成目标英文句子
    def decode(self, ids):
        tokens = [ self.id2word[id] for id in ids ]
        text = self.detokenizer.detokenize(tokens)
        return text

if __name__ == '__main__':
    # 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    print("中文词表大小：", zh_tokenizer.vocab_size)
    print("英文词表大小：", en_tokenizer.vocab_size)
    print(zh_tokenizer.eos_id)
    print(en_tokenizer.pad_id)

    text = "我爱自然语言处理。"

    ids = zh_tokenizer.encode(text)
    print(ids)

    text = "Hello world!"
    ids = en_tokenizer.encode(text, mark=True)
    print(ids)
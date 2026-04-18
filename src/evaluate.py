'''
    模型评估：BLEU
'''

import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

from config import *
from tokenizer import ZhTokenizer, EnTokenizer
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch

# 评估逻辑，返回BLEU值
def evaluate(model, test_loader, tgt_tokenizer, device):
    # 用列表保存所有预测结果和参考译文
    predictions = []
    references = []

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="评估"):
            inputs, targets = inputs.to(device), targets.tolist()
            # 1. 前向传播，得到预测结果
            batch_result = predict_batch(model, inputs, tgt_tokenizer, device)
            # 2. 将预测结果添加到列表
            predictions.extend(batch_result)
            # 3. 将参考译文添加到列表
            references.extend( [ [target[1:target.index(tgt_tokenizer.eos_id)]] for target in targets ] )

    return corpus_bleu(references, predictions)

# 评估主流程
def run_evaluate():
    # 1. 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 创建分词器
    zh_tokenizer = ZhTokenizer.from_vocab(MODELS_DIR / ZH_VOCAB_FILE)
    en_tokenizer = EnTokenizer.from_vocab(MODELS_DIR / EN_VOCAB_FILE)

    # 3. 加载模型
    model = TranslationModel(
        zh_tokenizer.vocab_size, en_tokenizer.vocab_size,
        zh_tokenizer.pad_id, en_tokenizer.pad_id,
    ).to(device)
    state_dict = torch.load(MODELS_DIR / BEST_MODEL)
    model.load_state_dict(state_dict)
    print("模型加载成功！")

    # 4. 创建测试集加载器
    test_loader = get_dataloader(train=False)

    # 5. 评估
    bleu = evaluate(model, test_loader, en_tokenizer, device)
    print("评估结果：BLEU-", bleu)

if __name__ == '__main__':
    run_evaluate()
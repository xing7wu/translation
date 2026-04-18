'''
    推理预测，应用程序
'''
import torch
from config import *
from model import TranslationModel
from tokenizer import ZhTokenizer, EnTokenizer

# 核心逻辑：传入模型输入(N, S)，批量预测
def predict_batch(model, inputs, tgt_tokenizer, device):
    model.eval()
    # 前向传播
    with torch.no_grad():
        # 1. 编码，得到三维张量(N, S, E)
        inputs = inputs.to(device)
        src_pad_mask = (inputs == model.src_padding_idx)
        memory = model.encode(inputs)

        # 2. 解码：自回归生成
        batch_size = inputs.shape[0]

        # 2.1 定义解码器初始输入(N, T=1)
        decoder_inputs = torch.full(size=(batch_size, 1), fill_value=tgt_tokenizer.sos_id).to(device)
        # 2.2 逐步生成下一个token，并添加到序列中循环迭代

        generated_tensors = []   # 将每一步生成的结果（N×1张量），保存到列表

        is_finished = torch.full(size=(batch_size,), fill_value=False).to(device)   # 记录每条数据是否已生成结束

        for i in range(MAX_SEQ_LEN):

            # 2.2.1 解码器前向传播，得到预测输出 (N, T, tgt_vocab_size)
            tgt_pad_mask = (decoder_inputs == model.tgt_padding_idx)
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_inputs.shape[1]).bool().to(device)
            decoder_outputs = model.decode(memory, memory_pad_mask=src_pad_mask, tgt=decoder_inputs,
                                           tgt_pad_mask=tgt_pad_mask, tgt_mask=tgt_mask)

            # 2.2.2 贪心解码，得到下一个token的id (N, 1)
            next_token_ids = torch.argmax(decoder_outputs[:, -1], dim=-1, keepdim=True)

            # 2.2.3 更新解码器输入，将新的token拼接在之前的序列后面
            decoder_inputs = torch.cat([decoder_inputs, next_token_ids], dim=-1)

            # 2.2.4 保存到列表
            generated_tensors.append(next_token_ids)

            # 2.2.5 判断是否结束
            is_finished |= (next_token_ids.squeeze(-1) == tgt_tokenizer.eos_id)
            if is_finished.all():
                break

        # 3. 自回归生成结束，处理生成句子
        # 3.1 合并生成结果，得到(N, L)的张量
        generated_token_ids = torch.cat(generated_tensors, dim=-1)
        # 3.2 转换为列表，方便截取
        generated_list = generated_token_ids.tolist()
        # 3.3 对每个句子进行截取，去掉<eos>及后面的所有token
        for i, sentence in enumerate(generated_list):
            # 判断如果有<eos>，就找到位置，进行截取
            if tgt_tokenizer.eos_id in sentence:
                eos_pos = sentence.index(tgt_tokenizer.eos_id)
                generated_list[i] = sentence[:eos_pos]

        return generated_list   # 返回真实 token id 构成的二维列表

# 预测主流程
def predict(text, model, src_tokenizer, tgt_tokenizer, device):
    # 1. 处理输入，张量 (N=1, L)
    ids = src_tokenizer.encode(text)
    input = torch.tensor([ids]).to(device)

    # 2. 调用核心预测逻辑
    result = predict_batch(model, input, tgt_tokenizer, device)

    # 3. 解码成英文句子返回
    return tgt_tokenizer.decode(result)

# 构建一个应用程序
def run_predict_app():
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
    state_dict = torch.load(MODELS_DIR/BEST_MODEL)
    model.load_state_dict(state_dict)
    print("模型加载成功！")

    # 4. 运行应用程序
    print("欢迎使用中英文翻译模型！请输入中文，输入q或者quit退出...")
    while True:
        user_input = input("中文：")
        if user_input in ['q', 'quit']:
            print("欢迎下次使用。")
            break
        if user_input.strip() == '':
            print("请输入有效内容：")
            continue
        # 核心预测逻辑
        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print("英文：", result)

if __name__ == '__main__':
    run_predict_app()

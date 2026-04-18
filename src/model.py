'''
    模型
'''
import math

import torch
import torch.nn as nn
from config import *

# 位置编码层
class PositionalEncoding(nn.Module):
    # 初始化，预计算位置编码矩阵
    def __init__(self, d_model, max_len=MAX_SEQ_LEN):
        super().__init__()
        # 定义位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 按公式逐个计算
        for pos in range(max_len):
            for _2i in range(0, d_model, 2):
                pe[pos, _2i] = math.sin(pos / (10000 ** (_2i / d_model)))
                pe[pos, _2i+1] = math.cos(pos / (10000 ** (_2i / d_model)))
        self.register_buffer("pe", pe)

    # 前向传播，传入词向量：三维张量(N, L, E)
    def forward(self, x):
        # 获取当前的序列长度L
        seq_len = x.shape[1]
        # 截取位置编码矩阵中，前L行
        part_pe = self.pe[0:seq_len]
        # 广播叠加
        return x + part_pe

# 自定义模型
class TranslationModel(nn.Module):
    # 初始化
    def __init__(self, src_vocab_size, tgt_vocab_size, src_padding_idx=0, tgt_padding_idx=0):
        super().__init__()
        self.src_padding_idx = src_padding_idx
        self.tgt_padding_idx = tgt_padding_idx
        # 定义词嵌入层
        self.src_embedding = nn.Embedding(src_vocab_size, DIM_MODEL, padding_idx=src_padding_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, DIM_MODEL, padding_idx=tgt_padding_idx)
        # 位置编码层
        self.pos_encoding = PositionalEncoding(DIM_MODEL, MAX_SEQ_LEN)
        # Transformer层
        self.transformer = nn.Transformer(
            d_model=DIM_MODEL,
            nhead=NUM_HEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS,
            batch_first=True
        )
        # 线性层（整合特征输出，映射到词表范围）
        self.linear = nn.Linear(DIM_MODEL, tgt_vocab_size)

    # 前向传播，传入src-(N, S), tgt-(N, T)
    def forward(self, src, tgt, src_pad_mask, tgt_mask, tgt_pad_mask):
        # 编码
        memory = self.encode(src, src_pad_mask)
        # 解码
        output = self.decode(tgt, memory, tgt_mask, tgt_pad_mask, memory_pad_mask=src_pad_mask)
        return output

    # 编码，输入 src-(N, S)
    def encode(self, src, src_pad_mask):
        # 1. 词嵌入，得到(N, S, E)
        embed = self.src_embedding(src)
        # 2. 位置编码
        input = self.pos_encoding(embed)
        # 3. Transformer编码器前向传播
        memory = self.transformer.encoder(src=input, src_key_padding_mask = src_pad_mask)
        return memory

    # 解码，输入 tgt-(N, T), memory-(N, S, E)
    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, memory_pad_mask):
        # 1. 词嵌入，得到(N, T, E)
        embed = self.tgt_embedding(tgt)
        # 2. 位置编码
        input = self.pos_encoding(embed)
        # 3. Transformer解码器前向传播，得到输出特征(N, T, E)
        output = self.transformer.decoder(
            tgt=input,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask
        )
        # 4. 线性层前向传播，映射到词表
        outputs = self.linear(output)
        return outputs

if __name__ == '__main__':
    # 定义参数
    batch_size = 32
    d_model = 64
    src_len = 10
    tgt_len = 15
    src_vocab_size = 1000  # 词表大小
    tgt_vocab_size = 1500

    # 定义数据
    src = torch.randint(src_vocab_size, (batch_size, src_len))
    tgt = torch.randint(tgt_vocab_size, (batch_size, tgt_len))

    # 定义模型
    model = TranslationModel(src_vocab_size, tgt_vocab_size)

    # 定义掩码
    padding_id = 0
    src_pad_mask = (src == padding_id)
    tgt_pad_mask = (tgt == padding_id)
    tgt_mask = model.transformer.generate_square_subsequent_mask(tgt_len).bool()

    # 前向传播
    output = model(src, tgt, src_pad_mask=src_pad_mask, tgt_mask=tgt_mask, tgt_pad_mask=tgt_pad_mask)

    print(output.shape)
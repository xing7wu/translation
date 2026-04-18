'''
    数据集和数据加载器
'''
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from torch.nn.utils.rnn import pad_sequence

# 自定义数据集类
class TranslationDataset(Dataset):
    # 初始化
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, orient='records', lines=True).to_dict(orient='records')

    # 获取数据集长度（大小）
    def __len__(self):
        return len(self.data)

    # 根据id获取元素（张量元组）
    def __getitem__(self, idx):
        input = torch.tensor(self.data[idx]['zh'])
        target = torch.tensor(self.data[idx]['en'])
        return input, target

# 定义对齐函数，传入参数batch，二元组列表
def collate_fn(batch):
    input_list = [ item[0] for item in batch ]
    target_list = [ item[1] for item in batch ]
    # 分别按最大长度填充
    input_batch = pad_sequence(input_list, batch_first=True)
    target_batch = pad_sequence(target_list, batch_first=True)
    return input_batch, target_batch

# 定义函数：获取数据加载器
def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    # 创建数据集
    dataset = TranslationDataset(path)
    # 创建加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=train, collate_fn=collate_fn)
    return dataloader

if __name__ == '__main__':
    # 创建数据集
    dataset = TranslationDataset(PROCESSED_DATA_DIR/TRAIN_DATA_FILE)
    print(len(dataset))
    print(dataset[0])

    # 创建加载器
    train_loader = get_dataloader(train=True)
    test_loader = get_dataloader(train=False)
    print(len(train_loader))
    print(len(test_loader))

    for input, target in train_loader:
        print(input.size(), target.size())
        break

    data_iter = iter(test_loader)
    input, target = next(data_iter)
    print(input.shape, target.shape)
# author: sunshine
# datetime:2021/7/2 下午2:06

# 数据处理器

from torch.utils.data import DataLoader, Dataset
from functools import partial
import torch
import json


def load_data(path, labels):
    """
    样本格式: (text, id)
    """
    D = []
    with open(path, 'r', encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            text, label = l['sentence'], l['label']
            D.append((text, labels.index(label)))
    return D


class NewsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    def create_collate_fn(self):
        def collate(examples):
            inputs = self.tokenizer([e[0] for e in examples], padding='longest', max_length=self.max_len,
                                    truncation='longest_first')

            input_ids = sum([[item, item] for item in inputs['input_ids']], [])
            attention_mask = sum([[item, item] for item in inputs['attention_mask']], [])
            token_type_ids = sum([[item, item] for item in inputs['token_type_ids']], [])

            # input_ids = torch.tensor(inputs['input_ids'], dtype=torch.long).repeat(2, 1)
            # attention_mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).repeat(2, 1)
            # token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).repeat(2, 1)

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)

            label = sum([[item, item] for item in [e[1] for e in examples]], [])
            label = torch.tensor(label, dtype=torch.long)

            return input_ids, attention_mask, token_type_ids, label

        return partial(collate)

    def get_data_loader(self, batch_size, shuffle=True, num_workers=0):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=self.create_collate_fn())


if __name__ == '__main__':
    a = [[1, 2, 3], [4, 5, 6]]

    b = [[i, i] for i in a]
    print(b)
    print(sum(b, []))

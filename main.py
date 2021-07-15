# author: sunshine
# datetime:2021/7/2 下午2:06
from argparse import Namespace
from train import Trainer
from data_loader import load_data, NewsDataset
from transformers import BertTokenizer
import json


def get_args():
    params = dict(
        max_len=128,
        batch_size=4,
        drop=0.3,
        epoch_num=10,
        learning_rate=2e-5,
        warmup_proportion=0.1,
        data_path='/home/sunshine/datasets/tnews/',
        output='output',
        bert_path='/home/sunshine/pre_models/pytorch/bert-base-chinese/',
        train_mode='train'
    )
    return Namespace(**params)


def build_dataset(args, tokenizer):
    """
    数据处理
    :return:
    """
    labels = [
        "100", "101", "102", "103", "104", "106", "107", "108", "109", "110", "112",
        "113", "114", "115", "116"
    ]

    train_data = load_data(args.data_path + '/train.json', labels)
    valid_data = load_data(args.data_path + '/dev.json', labels)
    print(len(train_data))
    train_loader = NewsDataset(train_data, tokenizer).get_data_loader(batch_size=args.batch_size, shuffle=True)
    valid_loader = NewsDataset(valid_data, tokenizer).get_data_loader(batch_size=args.batch_size, shuffle=False)
    print(len(train_loader))
    return [train_loader, valid_loader], labels


def main():
    # 准备参数
    args = get_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_path)

    # 处理数据
    data_loader, labels = build_dataset(args, tokenizer)

    # 构建trainer

    trainer = Trainer(
        args=args,
        data_loaders=data_loader,
        tokenizer=tokenizer,
        num_labels=len(labels)
    )

    trainer.train(args)


if __name__ == '__main__':
    main()
    # args = get_args()
    # tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    # a = tokenizer(['我是中国人', '张三十上班张三是水电费是否'], padding='longest', max_length=30, truncation='longest_first')
    # print(a)
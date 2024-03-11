import argparse
import random
import sys
import numpy as np
import pandas as pd
import math
from collections import namedtuple
from sklearn.model_selection import KFold, train_test_split, RepeatedKFold
from torch.utils.data import DataLoader


def load_data(input_file):
    dataset = pd.read_csv(input_file, sep='\t', index_col=None)
    dataset = dataset[dataset['halflife'] > 0]
    dataset = dataset[dataset['i'] > 0]
    dataset = dataset[
        dataset['p_history'].map(lambda x: len(str(x).split(','))) == dataset['t_history'].map(
            lambda x: len(x.split(',')))]
    dataset['weight_std'] = 1
    return dataset


if __name__ == "__main__":
    # 读取数据 拆分训练集和测试集
    dataset = load_data("./data/opensource_dataset_p_history_split.tsv")
    test = dataset.sample(frac=0.8, random_state=2024)
    train = dataset.drop(index=test.index)

    train_train, train_test = train_test_split(train, test_size=0.5, random_state=2022)
    sys.stderr.write('|train| = %d\n' % len(train_train))
    sys.stderr.write('|test|  = %d\n' % len(train_test))

    from model.Transformer_HLR import SpacedRepetitionModel

    model = SpacedRepetitionModel(train_train, train_test)
    model.train()
    model.eval(0, 0)

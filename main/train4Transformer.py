import argparse
import random
import sys
import numpy as np
import pandas as pd
import math
from collections import namedtuple
from sklearn.model_selection import train_test_split, RepeatedKFold

Instance = namedtuple('Instance', 'p t fv h a r_history t_history p_history'.split())

def load_data(input_file):
    dataset = pd.read_csv(input_file, sep='\t', index_col=None)
    dataset = dataset[dataset['halflife'] > 0]
    dataset = dataset[dataset['i'] > 0]
    dataset = dataset[
        dataset['p_history'].map(lambda x: len(str(x).split(','))) == dataset['t_history'].map(
            lambda x: len(x.split(',')))]
    # dataset.drop_duplicates(subset=['r_history', 't_history', 'p_history', 'difficulty'], inplace=True)
    # dataset['weight'] = dataset['total_cnt'] / dataset['total_cnt'].sum()
    # dataset['weight'] = dataset['total_cnt'] / dataset['total_cnt'].sum()
    # std = preprocessing.MinMaxScaler()
    # dataset['weight_std'] = std.fit_transform(dataset[['weight']]) + 1
    dataset['weight_std'] = 1
    return dataset


def feature_extract(train_set, test_set, method, omit_lexemes=False):
    instances = {'train': [], 'test': []}
    for set_id, data in (('train', train_set), ('test', test_set)):
        for i, row in data.iterrows():
            p = float(row['p_recall'])
            t = max(1, int(row['delta_t']))
            h = float(row['halflife'])
            right = row['r_history'].count('1')
            wrong = row['r_history'].count('0')
            total = right + wrong
            # feature vector is a list of (feature, value) tuples
            fv = []
            # core features based on method
            # optional flag features
            if method == 'pimsleur':
                fv.append((sys.intern('total'), right + wrong))
            elif method == 'leitner':
                fv.append((sys.intern('diff'), right - wrong))
            else:
                fv.append((sys.intern('right'), math.sqrt(1 + right)))
                fv.append((sys.intern('wrong'), math.sqrt(1 + wrong)))

            if method == 'LR':
                fv.append((sys.intern('time'), t))
            if not omit_lexemes:
                fv.append((sys.intern('%s' % (row['d'])), 1.))

            fv.append((sys.intern('bias'), 1.))
            instances[set_id].append(
                Instance(p, t, fv, h, (right + 2.) / (total + 4.), row['r_history'], row['t_history'],
                         row['p_history']))
            if i % 1000000 == 0:
                sys.stderr.write('%d...' % i)
        sys.stderr.write('done!\n')
    return instances['train'], instances['test']


# 定义参数
argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
argparser.add_argument('-p', action="store_true", default=False, help='omit p history features')
argparser.add_argument('-t', action="store_true", default=False, help='omit t history features')
argparser.add_argument('-test', action="store_true", default=False, help='test my_model')
argparser.add_argument('-train', action="store_true", default=False, help='train my_model')
argparser.add_argument('-m', action="store", dest="method", default='Transformer',
                       help="LSTM, HLR, LR, SM2,Transformer")  # 训练方法
argparser.add_argument('-hidden', action="store", dest="h", default='2', help="4, 8, 16, 32")  # 隐藏层数量
argparser.add_argument('-hidden_dim', action="store", dest="hd", default='256', help="512, 1024")  # 编码器输出数量
argparser.add_argument('-loss', action="store", dest="loss", default='sMAPE', help="MAPE, L1, MSE, sMAPE")  # 损失函数
argparser.add_argument('input_file', action="store", help='log file for training')

if __name__ == "__main__":
    random.seed(2024)
    args = argparser.parse_args()
    # 读取数据 拆分训练集和测试集
    # TODO
    dataset = load_data(args.input_file)
    test = dataset.sample(frac=0.8, random_state=2024)
    train = dataset.drop(index=test.index)
    sys.stderr.write(str(test.shape) + "\n")
    sys.stderr.write(str(train.shape) + "\n")
    # 根据传递的参数选择训练、测试方式
    if not args.train:
        if not args.test:
            sys.stderr.write("without test and train\n")
            train_train, train_test = train_test_split(train, test_size=0.5, random_state=2024)
            sys.stderr.write('|train| = %d\n' % len(train_train))
            sys.stderr.write('|test|  = %d\n' % len(train_test))
            # TODO
            if args.method == 'Transformer':
                from main.my_model.Transformer_HLR import SpacedRepetitionModel

                sys.stderr.write('method = "%s"\n' % args.method)
                sys.stderr.write(f'4 --> n_layers\n')
                sys.stderr.write(f'256 --> n_hidden_dim\n')
                sys.stderr.write(f'3 --> kernel_size\n')
                sys.stderr.write(f'{args.loss} --> loss\n')
                model = SpacedRepetitionModel(train_train, train_test, n_heads=8, hidden_dim=256, num_layers=4,
                                              kernel_size=3,
                                              omit_p_history=False, omit_t_history=False, loss=args.loss,
                                              network=args.method)
                model.train()
                model.eval(0, 0)
                #1
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4,hidden_dim=256, num_layers=2, kernel_size=3,
                #                               omit_p_history=False, omit_t_history=False,loss=args.loss, network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 2
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # sys.stderr.write('--> omit_p_history\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=2,
                #                               kernel_size=3,
                #                               omit_p_history=True, omit_t_history=False, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 3
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # sys.stderr.write('--> omit_t_history\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=2,
                #                               kernel_size=3,
                #                               omit_p_history=False, omit_t_history=True, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 4
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # sys.stderr.write('--> omit_p_history\n')
                # sys.stderr.write('--> omit_t_history\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=2,
                #                               kernel_size=3,
                #                               omit_p_history=True, omit_t_history=True, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 5
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # sys.stderr.write('--> omit_p_history\n')
                # sys.stderr.write('--> omit_t_history\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=2,
                #                               kernel_size=3,
                #                               omit_p_history=True, omit_t_history=True, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 6
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'1 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=2,
                #                               kernel_size=1,
                #                               omit_p_history=False, omit_t_history=False, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 7
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'4 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=4,
                #                               kernel_size=3,
                #                               omit_p_history=False, omit_t_history=False, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 8
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'8 --> n_layers\n')
                # sys.stderr.write(f'256 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=256, num_layers=8,
                #                               kernel_size=3,
                #                               omit_p_history=False, omit_t_history=False, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
                # # 9
                # sys.stderr.write('----------------------------------------------------------------')
                # sys.stderr.write('method = "%s"\n' % args.method)
                # sys.stderr.write(f'2 --> n_layers\n')
                # sys.stderr.write(f'128 --> n_hidden_dim\n')
                # sys.stderr.write(f'3 --> kernel_size\n')
                # sys.stderr.write(f'{args.loss} --> loss\n')
                # my_model = SpacedRepetitionModel(train_train, train_test, n_heads=4, hidden_dim=128, num_layers=2,
                #                               kernel_size=3,
                #                               omit_p_history=False, omit_t_history=False, loss=args.loss,
                #                               network=args.method)
                # my_model.train()
                # my_model.eval(0, 0)
        else:  # -test
            sys.stderr.write("-test")
            # kf = KFold(n_splits=5, shuffle=True, random_state=2022)
            kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2022)
            for idx, (train_index, test_fold) in enumerate(kf.split(test)):
                train_fold = dataset.iloc[train_index]
                test_fold = dataset.iloc[test_fold]
                repeat = idx // 2 + 1
                fold = idx % 2 + 1
                sys.stderr.write(f'Repeat {repeat}, Fold {fold}\n')
                sys.stderr.write(f'|train| = {len(train_index)}\n')
                sys.stderr.write(f'|test|  = {len(test_fold)}\n')
                # TODO
                if args.method == 'Transformer':
                    from main.my_model.Transformer_HLR import SpacedRepetitionModel

                    model = SpacedRepetitionModel(train_fold, test_fold, omit_p_history=args.p, omit_t_history=args.t,
                                                  hidden_dim=int(args.hd), loss=args.loss, network=args.method)

                    model.train()
                    model.eval(repeat, fold)
                else:
                    break
            test['pp'] = test['p_recall'].mean()
            print(test['p_recall'].mean())
            test['mae(p)'] = abs(test['pp'] - test['p_recall'])
            print("mae(p)", test['mae(p)'].mean())
            test['hh'] = np.log(test['pp']) / np.log(test['p_recall']) * test['delta_t']
            test['MAPE(h)'] = abs((test['hh'] - test['halflife']) / test['halflife'])
            print("MAPE(h)", test['MAPE(h)'].mean())
    else:  # -train
        print("-train")
        # train_train, train_test = train_test_split(dataset, test_size=0.2, random_state=2022)
        sys.stderr.write('|train| = %d\n' % len(dataset))
        # TODO
        if args.method == 'Transformer':
            from main.my_model.Transformer_HLR import SpacedRepetitionModel

            model = SpacedRepetitionModel(dataset, dataset, omit_p_history=args.p, omit_t_history=args.t,
                                          hidden_dim=int(args.hd), loss=args.loss, network=args.method)
            model.train()
import pandas as pd

if __name__ == "__main__":
    # 读取整个TSV文件
    file_path = './data/opensource_dataset_p_history.tsv'
    df = pd.read_csv(file_path, sep='\t')

    # 随机选择30%的数据
    subset_df = df.sample(frac=0.3, random_state=42)  # 设置random_state以保持随机性可复制

    # 保存拆分的部分到新文件
    output_file_path = './data/opensource_dataset_p_history_split.tsv'
    subset_df.to_csv(output_file_path, sep='\t', index=False)







# # 定义参数
# argparser = argparse.ArgumentParser(description='Fit a SpacedRepetitionModel to data.')
# argparser.add_argument('-test', action="store_true", default=False, help='test model')
# argparser.add_argument('-train', action="store_true", default=False, help='train model')
# argparser.add_argument('-m', action="store", dest="method", default='GRU', help="LSTM, HLR, LR, SM2")  # 训练方法
# argparser.add_argument('-hidden', action="store", dest="h", default='16', help="4, 8, 16, 32")  # 隐藏层数量
# argparser.add_argument('-loss', action="store", dest="loss", default='MAPE', help="MAPE, L1, MSE, sMAPE")  # 损失函数
# argparser.add_argument('input_file', action="store", help='log file for training')
#
# if __name__ == "__main__":
#     random.seed(2022)
#     args = argparser.parse_args()
#     sys.stderr.write('method = "%s"\n' % args.method)
#
#     # 读取数据 拆分训练集和测试集
#     dataset = load_data(args.input_file)
#     test = dataset.sample(frac=0.8, random_state=2022)
#     train = dataset.drop(index=test.index)
#     if not args.train:
#         if not args.test:
#             train_train, train_test = train_test_split(train, test_size=0.5, random_state=2022)
#             sys.stderr.write('|train| = %d\n' % len(train_train))
#             sys.stderr.write('|test|  = %d\n' % len(train_test))
#             if args.method in rnn_algo:
#                 from model.RNN_HLR import SpacedRepetitionModel
#
#                 model = SpacedRepetitionModel(train_train, train_test, omit_p_history=args.p, omit_t_history=args.t,
#                                               hidden_nums=int(args.h), loss=args.loss, network=args.method)
#                 model.train()
#                 model.eval(0, 0)
#         else:  # -train -test
#             # kf = KFold(n_splits=5, shuffle=True, random_state=2022)
#             kf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=2022)
#             for idx, (train_index, test_fold) in enumerate(kf.split(test)):
#                 train_fold = dataset.iloc[train_index]
#                 test_fold = dataset.iloc[test_fold]
#                 repeat = idx // 2 + 1
#                 fold = idx % 2 + 1
#                 sys.stderr.write('Repeat %d, Fold %d\n' % (repeat, fold))
#                 sys.stderr.write('|train| = %d\n' % len(train_index))
#                 sys.stderr.write('|test|  = %d\n' % len(test_fold))
#                 if args.method in rnn_algo:
#                     from model.RNN_HLR import SpacedRepetitionModel
#
#                     model = SpacedRepetitionModel(train_fold, test_fold, omit_p_history=args.p, omit_t_history=args.t,
#                                                   hidden_nums=int(args.h), loss=args.loss, network=args.method)
#                     model.train()
#                     model.eval(repeat, fold)
#
#             test['pp'] = test['p_recall'].mean()
#             print(test['p_recall'].mean())
#             test['mae(p)'] = abs(test['pp'] - test['p_recall'])
#             print("mae(p)", test['mae(p)'].mean())
#             test['hh'] = np.log(test['pp']) / np.log(test['p_recall']) * test['delta_t']
#             test['MAPE(h)'] = abs((test['hh'] - test['halflife']) / test['halflife'])
#             print("MAPE(h)", test['MAPE(h)'].mean())
#     else:  # -train
#         # train_train, train_test = train_test_split(dataset, test_size=0.2, random_state=2022)
#         sys.stderr.write('|train| = %d\n' % len(dataset))
#         if args.method in rnn_algo:
#             from model.RNN_HLR import SpacedRepetitionModel
#
#             model = SpacedRepetitionModel(dataset, dataset, omit_p_history=args.p, omit_t_history=args.t,
#                                           hidden_nums=int(args.h), loss=args.loss, network=args.method)
#             model.train()
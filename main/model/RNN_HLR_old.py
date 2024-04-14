import torch.optim.lr_scheduler
import torch.nn as nn
import numpy as np
import torch.nn.functional
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
from pathlib import Path
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def l1_loss(forecast, actual):
    return nn.functional.l1_loss(forecast, actual)


def mse_loss(forecast, actual):
    return nn.functional.mse_loss(forecast, actual)


def mape_loss(forecast, actual):
    return nn.functional.l1_loss(forecast, actual) / abs(actual)


def smape_loss(forecast, actual):
    return nn.functional.l1_loss(forecast, actual) / (abs(forecast) + abs(actual)) * 2


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class RNN(nn.Module):
    def __init__(self, n_letters, n_hidden, n_categories, network, attention_flag = True):
        super().__init__()
        self.n_input = n_letters
        self.n_hidden = n_hidden
        self.n_out = n_categories
        self.attention_flag = attention_flag
        if attention_flag:
            self.attention = nn.MultiheadAttention(embed_dim=self.n_hidden, num_heads=1)

        if network == 'GRU':
            self.rnn = nn.GRU(self.n_input, self.n_hidden, 1)
        elif network == "LSTM":
            self.rnn = nn.LSTM(self.n_input, self.n_hidden, 1)
        else:
            self.rnn = nn.RNN(self.n_input, self.n_hidden, 1)
        self.fc = nn.Linear(self.n_hidden, self.n_out)

    def forward(self, x, hx):
        if self.attention_flag:
            x, h = self.rnn(x, hx=hx)
            attention_output, _ = self.attention(x, x, x)  # 注意力机制
            x = x + attention_output
        else:
            x, h = self.rnn(x, hx=hx)
        output = torch.exp(self.fc(x[-1]))
        return output, h

    def full_connect(self, h):
        return self.fc(h)


class SpacedRepetitionModel(object):
    def __init__(self, train_set, test_set, omit_p_history=False, omit_t_history=False, hidden_nums=16, loss="MAPE",
                 network="GRU"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置GPU加速
        print('device:', self.device)
        self.attention_flag = False
        self.n_hidden = hidden_nums  # 隐藏层数量
        self.omit_p = omit_p_history  # 是否省略p_history
        self.omit_t = omit_t_history  # 是否省略t_history
        if omit_p_history and omit_t_history:
            self.feature_num = 1
        elif omit_p_history or omit_t_history:
            self.feature_num = 2
        else:
            self.feature_num = 3
        self.net_name = network  # 使用的网络
        self.net = RNN(self.feature_num, self.n_hidden, 1, network, self.attention_flag).to(device=self.device)
        self.lr = 1e-3  # 学习率
        self.weight_decay = 1e-5  # 权重衰减 正则化
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)  # Adam优化器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=32)  # 余弦退火学习率调度器

        self.current_loss = 0
        self.avg_train_losses = []
        self.avg_eval_losses = []
        self.train_set = train_set
        self.test_set = test_set
        self.train_cnt = len(train_set)
        self.n_iter = 1000000
        self.n_epoch = int(self.n_iter / self.train_cnt + 1)  # 定义迭代次数
        self.print_every = int(self.train_cnt / 4)  # 每一个epoch汇报4次进度
        self.plot_every = self.train_cnt  # 每一个epoch汇报1次进度
        self.loss_name = loss  # 损失函数
        if loss == "MAPE":
            self.loss = mape_loss
        elif loss == "L1":
            self.loss = l1_loss
        elif loss == "MSE":
            self.loss = mse_loss
        elif loss == "sMAPE":
            self.loss = smape_loss

        self.writer = SummaryWriter(comment=self.write_title())  # SummaryWriter 是 PyTorch 提供的用于将信息写入 TensorBoard 的工具

    def write_title(self):
        title = f'nn-{self.net_name}_nh-{self.n_hidden}_loss-{self.loss_name}'
        if self.omit_p:
            title += "-p"
        if self.omit_t:
            title += "-t"
        if self.attention_flag:
            title += "-atten"
        return title

    def train(self):
        start = time.time()
        for i in range(self.n_epoch):
            print(f"Epoch: {i + 1}")
            train_set = shuffle(self.train_set, random_state=i)
            for idx, index in enumerate(train_set.index):
                # 半衰期、历史特征列表、半衰期张量、历史记录张量、样本的权重标准化信息
                halflife, line, halflife_tensor, line_tensor, weight = self.sample2tensor(train_set.loc[index])
                line_tensor = line_tensor.to(device=self.device)
                halflife_tensor = halflife_tensor.to(device=self.device)
                self.net.train()
                self.optimizer.zero_grad()  # 梯度清为0
                output, _ = self.net(line_tensor, None)
                loss = self.loss(output[0], halflife_tensor) * weight
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                # self.current_loss += loss.data.item()

                iterations = idx + i * self.train_cnt + 1

                if iterations % self.print_every == 0:
                    tmp = output.cpu()
                    guess = tmp.detach().numpy()
                    print(guess)
                    correct = halflife_tensor[0]
                    print(
                        '%d %d%% (%s) %.4f %.4f %.4f / %s' % (
                            iterations, iterations / (self.n_epoch * self.train_cnt) * 100, time_since(start),
                            loss.data.item(), guess[-1], correct, line))

                if iterations % self.plot_every == 0:
                    self.net.eval()
                    for stage in ('train', 'test'):
                        if stage == 'train':
                            dataset = self.train_set
                        else:
                            dataset = self.test_set
                        plot_loss = 0
                        plot_count = 0
                        with torch.no_grad():
                            for plot_index in dataset.index:
                                halflife, line, halflife_tensor, line_tensor, weight = self.sample2tensor(
                                    dataset.loc[plot_index])
                                line_tensor = line_tensor.to(device=self.device)
                                halflife_tensor = halflife_tensor.to(device=self.device)
                                output, _ = self.net(line_tensor, None)
                                loss = self.loss(output[0], halflife_tensor) * weight
                                plot_loss += loss.data.item()
                                plot_count += 1
                        if stage == 'train':
                            self.avg_train_losses.append(plot_loss / plot_count)
                        else:
                            self.avg_eval_losses.append(plot_loss / plot_count)

                    print('Iteration %d %d%% (%s); Avg_train_loss %.4f; Avg_eval_loss %.4f' % (
                        iterations, iterations / (self.n_epoch * self.train_cnt) * 100, time_since(start),
                        self.avg_train_losses[-1],
                        self.avg_eval_losses[-1]))

            self.writer.add_scalar('train_loss', self.avg_train_losses[-1], i + 1)
            self.writer.add_scalar('eval_loss', self.avg_eval_losses[-1], i + 1)

        title = self.write_title()
        self.net.eval()
        # 保存模型
        path = f'./tmp/{title}'
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.net, f'{path}/model.pth')
        # example_input = torch.rand(1, 1, self.feature_num)
        # if self.net_name == "LSTM":
        #     example_hidden = (torch.randn(1, 1, self.n_hidden), torch.randn(1, 1, self.n_hidden))
        #     fully_traced = torch.jit.trace_module(self.net, {'forward': (example_input, example_hidden)})
        # else:
        #     example_hidden = torch.rand(1, 1, self.n_hidden)
        #     fully_traced = torch.jit.trace_module(self.net, {'forward': (example_input, example_hidden),
        #                                                  'full_connect': example_hidden})
        # fully_traced.save(f'{path}/model.pt')
        # self.writer.add_graph(self.net, [example_input, example_hidden])
        self.writer.close()

    def eval(self, repeat, fold):
        record = pd.DataFrame(
            columns=['r_history', 't_history', 'p_history','t','halflife', 'hh', 'p', 'pp', 'ae', 'ape'])
        self.net.eval()
        with torch.no_grad():
            ae = 0
            ape = 0
            count = 0
            for index in tqdm(self.test_set.index):
                sample = self.test_set.loc[index]
                halflife, line, halflife_tensor, line_tensor, weight = self.sample2tensor(sample)
                line_tensor = line_tensor.to(device=self.device)
                halflife_tensor = halflife_tensor.to(device=self.device)
                output = self.net(line_tensor, None)
                output = float(output[0])
                pp = np.exp(np.log(0.5) * sample['delta_t'] / output)
                p = sample['p_recall']
                ae += abs(p - pp)
                ape += abs(output - sample['halflife']) / sample['halflife']
                count += 1

                record = pd.concat([record, pd.DataFrame(
                    {'r_history': [sample['r_history']],
                     't_history': [sample['t_history']],
                     'p_history': [sample['p_history']],
                     't': [sample['delta_t']], 'h': [sample['halflife']],
                     'hh': [round(output, 2)], 'p': [sample['p_recall']],
                     'pp': [round(pp, 3)], 'ae': [round(abs(p - pp), 3)],
                     'ape': [round(abs(output - sample['halflife']) / sample['halflife'], 3)]})],
                                   ignore_index=True)
            print(f"model: gru")
            print(f'sample num: {count}')
            print(f"mae: {ae / count}")
            print(f"mape: {ape / count}")
            title = self.write_title()
            path = f'./result/{title}'
            Path(path).mkdir(parents=True, exist_ok=True)
            record.to_csv(f'{path}/repeat{repeat}_fold{fold}_{int(time.time())}.tsv', index=False, sep='\t')

    # 将样本转换为张量
    def sample2tensor(self, sample):
        halflife = sample['halflife']
        features = [sample['r_history'], sample['t_history'], sample['p_history']]
        r_history = sample['r_history'].split(',')
        t_history = sample['t_history'].split(',')
        p_history = sample['p_history'].split(',')
        sample_tensor = torch.zeros(len(r_history), 1, self.feature_num)  # 初始化为一个三维张量

        for li, response in enumerate(r_history):
            sample_tensor[li][0][0] = int(response)
            # 根据是否省略 p 或 t 特征，选择填充相应的历史数据
            if self.omit_p and self.omit_t:
                continue
            elif self.omit_t and not self.omit_p:
                sample_tensor[li][0][1] = float(p_history[li])
            elif self.omit_p and not self.omit_t:
                sample_tensor[li][0][1] = float(t_history[li])
            else:
                sample_tensor[li][0][1] = float(t_history[li])
                sample_tensor[li][0][2] = float(p_history[li])

        halflife_tensor = torch.tensor([halflife], dtype=torch.float32)
        # 返回半衰期、历史特征列表、半衰期张量、历史记录张量和样本的权重标准化信息
        return halflife, features, halflife_tensor, sample_tensor, sample[['weight_std']].values[0]
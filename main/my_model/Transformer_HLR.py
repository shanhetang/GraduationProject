import pandas as pd
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
import numpy as np
import torch.nn.functional as F
import time
import math
from pathlib import Path
from sklearn.utils import shuffle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


# l1_loss:平均绝对误差MAE
def l1_loss(forecast, actual):
    return F.l1_loss(forecast, actual)


def mse_loss(forecast, actual):
    return F.mse_loss(forecast, actual)


def mape_loss(forecast, actual):
    return F.l1_loss(forecast, actual) / abs(actual)


def smape_loss(forecast, actual):
    return F.l1_loss(forecast, actual) / (abs(forecast) + abs(actual)) * 2


class CausalConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1, ):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0, )

        self.__padding = kernel_size - 1

    def forward(self, x):
        """
        Inputs of forward function
        Shape:
            x: [batch size, sequence length, feat_dim]
            output: [batch size, sequence length, feat_dim]
        """
        x = x.permute(0, 2, 1)
        x = super(CausalConv1d, self).forward(F.pad(x, (self.__padding, 0)))
        return x.permute(0, 2, 1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self,
                 d_model,
                 dropout,
                 max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    def __init__(self,
                 d_model,
                 dropout=0.1,
                 max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.permute(1, 0, 2)


class TransformerModel(nn.Module):
    def __init__(self,
                 input_dim,
                 d_model,
                 num_heads,
                 encode_layers_num,
                 output_dim=1,
                 dropout=0.1,
                 kernel_size=3,
                 pos='learnable'):
        """
        :param input_dim: 输入的特征数量
        :param d_model: 时间步嵌入
        :param output_dim: 输出的维度
        """
        super(TransformerModel, self).__init__()
        # 因果卷积层
        self.conv = CausalConv1d(in_channels=input_dim, out_channels=d_model, kernel_size=kernel_size)

        # 定义位置编码器
        if pos == 'learnable':
            self.positional_encoding = LearnablePositionalEncoding(d_model, dropout=dropout)
        elif pos == 'traditional':
            self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        self.encode_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout,
                                                       batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encode_layer, num_layers=encode_layers_num)

        # decoder 映射到输出维度
        self.decoder = nn.Linear(d_model, output_dim)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, sample_tensor):
        """
        shape
         sample_tensor: [batch size, sequence length, feat_dim]
            output: [batch size, sequence length,  feat_dim]
        """
        t_src = self.conv(sample_tensor)

        t_src = self.positional_encoding(t_src)

        t_output = self.transformer_encoder(src=t_src)

        output = self.decoder(t_output)
        return output


class SpacedRepetitionModel(object):
    def __init__(self,
                 train_set,
                 test_set,
                 n_heads,
                 d_model=256,
                 num_layers=2,
                 kernel_size=3,
                 omit_p_history=False,
                 omit_t_history=False,
                 loss="sMAPE",
                 network="Transformer"):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设置GPU加速
        print('device:', self.device)

        self.omit_p = omit_p_history  # 是否省略p_history
        self.omit_t = omit_t_history  # 是否省略t_history
        if omit_p_history and omit_t_history:
            self.feature_num = 1
        elif omit_p_history or omit_t_history:
            self.feature_num = 2
        else:
            self.feature_num = 3

        # 模型初始化
        self.net_name = network  # 使用的网络
        self.d_model = d_model  # 编码器解码器特征数量
        self.num_heads = n_heads  # 多头自注意力机制的头数
        self.num_layers = num_layers  # Transformer encoder层数
        self.kernel_size = kernel_size  # 卷积的核数
        self.net = TransformerModel(self.feature_num, self.d_model, self.num_heads,
                                    self.num_layers, kernel_size=self.kernel_size).to(self.device)

        # 损失函数
        self.loss_name = loss
        if loss == "MAPE":
            self.loss = mape_loss
        elif loss == "L1":
            self.loss = l1_loss
        elif loss == "MSE":
            self.loss = mse_loss
        elif loss == "sMAPE":
            self.loss = smape_loss

        self.lr = 1e-5  # 学习率
        self.weight_decay = 1e-3  # 权重衰减 正则化
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
        print(f"Epoch num: {self.n_epoch}")
        self.print_every = int(self.train_cnt / 4)  # 每一个epoch汇报4次进度
        self.plot_every = self.train_cnt  # 每一个epoch汇报1次进度

        # SummaryWriter是PyTorch提供的用于将信息写入TensorBoard的工具
        self.writer = SummaryWriter(comment=self.write_title())

    def write_title(self):
        title = f'{self.net_name}-d_model={self.d_model}-nhead={self.num_heads}-encoder_num={self.num_layers}' \
                f'-loss={self.loss_name}'
        if self.kernel_size != 3:
            title += "-k=" + str(self.kernel_size)
        if self.omit_p:
            title += "-p"
        if self.omit_t:
            title += "-t"
        return title

    def train(self):
        start = time.time()
        for i in range(self.n_epoch):
            print(f"Epoch: {i + 1}")
            train_set = shuffle(self.train_set, random_state=i)
            for idx, index in enumerate(train_set.index):
                self.net.train()
                self.optimizer.zero_grad()  # 梯度清为0
                # 半衰期、历史特征列表、半衰期张量、历史记录张量
                halflife, line, halflife_tensor, line_tensor = self.sample2tensor(train_set.loc[index])
                line_tensor = line_tensor.to(device=self.device)
                halflife_tensor = halflife_tensor.to(device=self.device)
                output = self.net(line_tensor)
                loss = self.loss(output[:, -1, 0], halflife_tensor)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                iterations = idx + i * self.train_cnt + 1

                if iterations % self.print_every == 0:
                    tmp = output.cpu()
                    guess = tmp.detach().numpy()
                    correct = halflife_tensor[0]
                    print('%d %d%% (%s) %.4f %.4f %.4f / %s' % (
                        iterations, iterations / (self.n_epoch * self.train_cnt) * 100, time_since(start),
                        loss.data.item(), guess[:, -1, :], correct,
                        line))

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
                                halflife, line, halflife_tensor, line_tensor = self.sample2tensor(
                                    dataset.loc[plot_index])
                                line_tensor = line_tensor.to(device=self.device)
                                halflife_tensor = halflife_tensor.to(device=self.device)
                                output = self.net(line_tensor)
                                loss = self.loss(output[:, -1, 0], halflife_tensor)
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

    def eval(self, repeat, fold):
        record = pd.DataFrame(
            columns=['r_history', 't_history', 'p_history',
                     't', 'halflife', 'hh', 'p', 'pp', 'ae', 'ape'])
        self.net.eval()
        with torch.no_grad():
            ae = 0
            ape = 0
            count = 0
            for index in tqdm(self.test_set.index):
                sample = self.test_set.loc[index]
                halflife, line, halflife_tensor, line_tensor = self.sample2tensor(sample)
                line_tensor = line_tensor.to(device=self.device)
                halflife_tensor = halflife_tensor.to(device=self.device)
                output = self.net(line_tensor)
                output = output.cpu()
                output = float(output[:, -1, 0])

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
            print(f"model: transformer")
            print(f'sample num: {count}')
            print(f"mae: {ae / count}")
            print(f"mape: {ape / count}")
            title = self.write_title()
            path = f'./result/{title}'
            Path(path).mkdir(parents=True, exist_ok=True)
            record.to_csv(f'{path}/repeat{repeat}_fold{fold}_{int(time.time())}.tsv', index=False, sep='\t')

    def sample2tensor(self, sample):
        halflife = sample['halflife']
        features = [sample['r_history'], sample['t_history'], sample['p_history']]
        r_history = sample['r_history'].split(',')
        t_history = sample['t_history'].split(',')
        p_history = sample['p_history'].split(',')

        halflife_tensor = torch.tensor([halflife], dtype=torch.float32)
        sample_tensor = torch.zeros(1, len(r_history), self.feature_num)
        for li, response in enumerate(r_history):
            sample_tensor[0][li][0] = int(response)
            # 根据是否省略 p 或 t 特征，选择填充相应的历史数据
            if self.omit_p and self.omit_t:
                continue
            elif self.omit_t and not self.omit_p:
                sample_tensor[0][li][1] = float(p_history[li])
            elif self.omit_p and not self.omit_t:
                sample_tensor[0][li][1] = float(t_history[li])
            else:
                sample_tensor[0][li][1] = float(t_history[li])
                sample_tensor[0][li][2] = float(p_history[li])

        # 返回半衰期、历史特征列表、半衰期张量、历史记录张量
        return halflife, features, halflife_tensor, sample_tensor,

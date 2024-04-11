# -*- Coding: UTF-8 -*-
# Author: Answer   Time:2021/2/15
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.manual_seed(0)
np.random.seed(0)

# This concept is also called teacher forceing.
# The flag decides if the loss will be calculted over all
# or just the predicted values.
calculate_loss_over_all_values = False

input_window = 50
output_window = 5
batch_size = 50  # batch size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#### 数据预处理 ####
def get_padding_mask(input):
    input1 = pd.DataFrame(input)
    for pub in range(1970, 2020):
        if pub == 1970:
            input1.loc[input1['出版年'] == 1970, '1970':'2019'] = 0
        else:
            input1.loc[input1['出版年'] == 1970, '1970':str(pub - 1)] = 1  # 1 转换为 bool 是 TRUE，表示掩码
            input1.loc[input1['出版年'] == 1970, str(pub):'2019'] = 0
    input1 = input1.loc[:, '1970':'2019']
    return input1


def get_data():
    data = pd.read_csv('cleaned_data.csv', encoding='GBK', chunksize=1000)
    data = pd.concat(data, ignore_index=True)
    data = data.dropna()
    for i in range(1971, 2020):
        data[str(i)] = data[str(i - 1)] + data[str(i)]
    data = data[data['出版年'] <= 2010]
    data = data[data['2015'] > 0]
    data.reset_index(inplace=True, drop=True)
    data = data.loc[0:3001, :]
    train_data, test_data = train_test_split(data, train_size=0.8, random_state=10)
    data.reset_index(inplace=True, drop=True)
    train_data.reset_index(inplace=True, drop=True)
    test_data.reset_index(inplace=True, drop=True)
    train_data1 = train_data
    test_data1 = test_data
    train_data = train_data.loc[:, '1970':'2019']
    test_data = test_data.loc[:, '1970':'2019']
    max1 = max(max(test_data.max()), max(train_data.max()))
    train_data = train_data / max1
    test_data = test_data / max1

    train_padding = get_padding_mask(train_data1)
    test_padding = get_padding_mask(test_data1)
    train_seq = torch.from_numpy(np.array(train_data.iloc[:, :-output_window]))
    train_label = torch.from_numpy(np.array(train_data.iloc[:, output_window:]))

    test_seq = torch.from_numpy(np.array(test_data.iloc[:, :-output_window]))
    test_label = torch.from_numpy(np.array(test_data.iloc[:, output_window:]))

    train_padding = torch.from_numpy(np.array(train_padding.iloc[:, :-output_window]))
    test_padding = torch.from_numpy(np.array(test_padding.iloc[:, :-output_window]))

    train_sequence = torch.stack((train_seq, train_label, train_padding), dim=1).type(torch.FloatTensor)
    test_data = torch.stack((test_seq, test_label, test_padding), dim=1).type(torch.FloatTensor)
    return train_sequence.to(device), test_data.to(device), max1


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    # seq_len = min(batch_size, len(source) - 1 - i)
    # data = source[i:i+seq_len]
    input = torch.stack(
        torch.stack([item[0] for item in data]).chunk(input_window, 1)
    )  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    padding = torch.stack(torch.stack([item[2] for item in data]).chunk(input_window, 1))
    padding = padding.squeeze(2).transpose(0, 1)
    # p = 0
    # for i in range(padding.size(0)):
    #     if torch.sum(padding[i]) == input.size(1):
    #         continue
    #     else:
    #         p = i
    #         break
    # input = input[p:]
    # target = target[p:]
    # padding = padding[p:]

    return input, target, padding


#### positional encoding ####
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


#### model stracture ####
class TransAm(nn.Module):
    def __init__(self, feature_size=512, num_layers=1, dropout=0):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()
        self.src_key_padding_mask = None

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_padding):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask
        if self.src_key_padding_mask is None:
            mask_key = src_padding.bool()
            self.src_key_padding_mask = mask_key

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask, self.src_key_padding_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output


def train(train_data):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets, key_padding_mask = get_batch(train_data, i, batch_size)
        optimizer.zero_grad()
        output = model(data, key_padding_mask)

        if calculate_loss_over_all_values:
            loss = criterion(output, targets)
        else:
            loss = criterion(output[-output_window:], targets[-output_window:])

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss))  # , math.exp(cur_loss)
            total_loss = 0
            start_time = time.time()


def plot_and_loss(eval_model, data_source, epoch):
    eval_model.eval()
    total_loss = 0.
    eval_batch_size = 50
    test_result = torch.Tensor(0)
    truth = torch.Tensor(0)
    test_result1 = torch.Tensor(0)
    truth1 = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, target, key_padding_mask = get_batch(data_source, i, eval_batch_size)
            # look like the model returns static values for the output window
            output = eval_model(data, key_padding_mask)
            if calculate_loss_over_all_values:
                total_loss += criterion(output, target).item()
            else:
                total_loss += criterion(output[-output_window:], target[-output_window:]).item()

            test_result = torch.cat((test_result, output[-1].squeeze(1).view(-1).cpu() * max1),
                                    0)  # todo: check this. -> looks good to me
            truth = torch.cat((truth, target[-1].squeeze(1).view(-1).cpu() * max1), 0)
            test_result1 = torch.cat((test_result1, output[-5:].squeeze(2).transpose(0, 1).cpu() * max1),
                                     0)  # todo: check this. -> looks good to me
            # test_result1 = pd.concat([test_result1,pd.DataFrame(output[-5:].squeeze(1).view(-1).cpu())], axis=0)
            # truth1 = pd.concat([truth1, pd.DataFrame(target[-5:].squeeze(1).view(-1).cpu())], axis=0)
            truth1 = torch.cat((truth1, target[-5:].squeeze(2).transpose(0, 1).cpu() * max1), 0)
    # test_result = test_result.cpu().numpy()
    # pyplot.plot(test_result[-output_window:], color="red")
    # pyplot.plot(truth[-output_window:], color="blue")
    # pyplot.plot(test_result[-output_window:] - truth[-output_window:], color="green")
    #     print(test_result)
    #     print(truth)
    pyplot.plot(torch.round(truth), color="blue", alpha=0.5)
    pyplot.plot(torch.round(test_result), color="red", alpha=0.5)
    pyplot.plot(torch.round(test_result - truth), color="green", alpha=0.8)
    pyplot.grid(True, which='both')
    pyplot.axhline(y=0, color='k')
    # pyplot.ylim((-2, 2))
    pyplot.savefig('epo%d.png' % epoch)
    pyplot.close()

    return total_loss / i, test_result, truth, torch.round(test_result1), torch.round(truth1)



def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 50
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets, key_padding_mask = get_batch(data_source, i, eval_batch_size)
            output = eval_model(data, key_padding_mask)
            if calculate_loss_over_all_values:
                total_loss += len(data[0]) * criterion(output, targets).cpu().item()
            else:
                total_loss += len(data[0]) * criterion(output[-output_window:], targets[-output_window:]).cpu().item()
    return total_loss / len(data_source)


train_data, val_data, max1 = get_data()
model = TransAm().to(device)

criterion = nn.MSELoss()
lr = 0.00001
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.96)

best_val_loss = float("inf")
epochs = 200  # The number of epochs
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(train_data)
    train_loss = evaluate(model, train_data)

    if (epoch % 5 is 0):
        val_loss, tran_output, tran_true, tran_output5, tran_true5 = plot_and_loss(model, val_data, epoch)
        # predict_future(model, val_data, 200)
    else:
        val_loss = evaluate(model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | train loss {:5.5f} '.format(
        epoch, (time.time() - epoch_start_time),
        val_loss, train_loss))  # , math.exp(val_loss) | valid ppl {:8.2f}
    print('-' * 89)
    scheduler.step()

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
# print(out)
# print(out.shape)

# t1, t2, t3 = get_batch(train_data, 0, 1)
# out = model(t1, t3)
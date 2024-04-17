import math
import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import tqdm

from model.utils import *
from envrioment import DHP_HLR, GRU_HLR

# plt.style.use('whitegrid')
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=1.5, y=1.5, z=1.25)
)
plt.rcParams["font.family"] = "SimHei"  # 处理中文无法正常显示的问题
plt.rcParams['axes.unicode_minus'] = False  # 负号显示


def concat_data(folder_path):
    # 初始化一个空列表，用于存储读取的数据框
    dfs = []
    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(f'./result/{folder_path}'):
        # 确保文件以 '.tsv' 结尾且不以 'repeat0_fold0' 开头
        if filename.endswith('.tsv') and not filename.startswith('repeat0_fold0'):
            # 构建文件的完整路径
            file_path = os.path.join('./result/', folder_path, filename)
            # 读取 TSV 文件，并将其存储到数据框中
            df = pd.read_csv(file_path, sep='\t')
            # 将数据框存储到列表中
            dfs.append(df)

    # 合并所有数据框
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df['halflife'] = np.log(merged_df['h'])
    merged_df['sape'] = round(abs(merged_df['h'] - merged_df['hh']) / (merged_df['h'] + merged_df['hh']), 3)
    return merged_df


def pre_halflive_visualize():
    # 绘制半衰期的分布
    plt.figure(figsize=(10, 10))  # 绘制召回概率的直方图
    ax = plt.subplot(111)
    raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
    counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    ax.set_xlabel('半衰期（h）\n(a)', fontsize=28)
    ax.set_ylabel('样本数量\n', fontsize=28)
    ax.tick_params(axis='both', labelsize=22)
    ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
    ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
                  ['1', '3', '10', '30', '100', '1000'])
    ax.set_axisbelow(True)  # 网格显现在图形下方
    # 绘制不同sMAPE的误差分布
    ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
    ax2.set_ylabel('\nsMAPE', fontsize=28)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    files = [
        'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
        'static-Transformer-d_model=256-nhead=4-encoder_num=4',
        '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
        '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',
        'DHP', 'HLR', 'HLR-lex']
    colors = [
        'green', 'purple',
        'orange', 'yellow', 'brown', 'cyan',
        'pink', 'magenta', 'lime']
    labels = [
        'Transformer-HLR','Transformer-HLR+d',
        'GRU-HLR', 'GRU&Attention-HLR',
        'GRU-HLR-t', 'GRU&Attention-HLR -t',
        'DHP-HLR', 'HLR', 'HLR-lex']
    for filename, color, label in zip(files, colors, labels):
        raw = concat_data(filename)
        raw['bin'] = pd.cut(raw['halflife'], bins=bins)
        bin_means = raw.groupby('bin', observed=True)['sape'].mean()
        ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
    # 添加图例
    ax2.legend(loc='upper right', fontsize=14)

    # ax = plt.subplot(132)
    # raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
    # counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax.set_xlabel('半衰期（h）\n(b)', fontsize=28)
    # ax.set_ylabel('样本数量\n', fontsize=28)
    # ax.tick_params(axis='both', labelsize=22)
    # ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
    # ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    # ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
    #               ['1', '3', '10', '30', '100', '1000'])
    # ax.set_axisbelow(True)  # 网格显现在图形下方
    # # 绘制不同sMAPE的误差分布
    # ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
    # ax2.set_ylabel('\nsMAPE', fontsize=28)
    # ax2.tick_params(axis='y', labelsize=22)
    # ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    # files = [
    #     'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
    #     'static-Transformer-d_model=256-nhead=4-encoder_num=4',]
    #     # '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
    #     # '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',
    #     # 'DHP', 'HLR', 'HLR-lex']
    # colors = [
    #     'green', 'purple',]
    #     # 'orange', 'yellow', 'brown', 'cyan',
    #     # 'pink', 'magenta', 'lime']
    # labels = [
    #     'Transformer-HLR', 'Transformer-HLR+d',]
    #     # 'GRU-HLR', 'GRU&Attention-HLR',
    #     # 'GRU-HLR-t', 'GRU&Attention-HLR -t',
    #     # 'DHP-HLR', 'HLR', 'HLR-lex']
    # for filename, color, label in zip(files, colors, labels):
    #     raw = concat_data(filename)
    #     raw['bin'] = pd.cut(raw['halflife'], bins=bins)
    #     bin_means = raw.groupby('bin', observed=True)['sape'].mean()
    #     ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
    # # 添加图例
    # ax2.legend(loc='upper right', fontsize=14)
    #
    # ax = plt.subplot(133)
    # raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
    # counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
    # bin_centers = (bins[:-1] + bins[1:]) / 2
    # ax.set_xlabel('半衰期（h）\n(c)', fontsize=28)
    # ax.set_ylabel('样本数量\n', fontsize=28)
    # ax.tick_params(axis='both', labelsize=22)
    # ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
    # ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    # ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
    #               ['1', '3', '10', '30', '100', '1000'])
    # ax.set_axisbelow(True)  # 网格显现在图形下方
    # # 绘制不同sMAPE的误差分布
    # ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
    # ax2.set_ylabel('\nsMAPE', fontsize=28)
    # ax2.tick_params(axis='y', labelsize=22)
    # ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    # files = [
    #     # 'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
    #     # 'static-Transformer-d_model=256-nhead=4-encoder_num=4', ]
    # '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
    # '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',]
    # # 'DHP', 'HLR', 'HLR-lex']
    # colors = [
    #     # 'green', 'purple', ]
    # 'orange', 'yellow', 'brown', 'cyan',]
    # # 'pink', 'magenta', 'lime']
    # labels = [
    #     # 'Transformer-HLR', 'Transformer-HLR+d', ]
    # 'GRU-HLR', 'GRU&Attention-HLR',
    # 'GRU-HLR-t', 'GRU&Attention-HLR -t',]
    # # 'DHP-HLR', 'HLR', 'HLR-lex']
    # for filename, color, label in zip(files, colors, labels):
    #     raw = concat_data(filename)
    #     raw['bin'] = pd.cut(raw['halflife'], bins=bins)
    #     bin_means = raw.groupby('bin', observed=True)['sape'].mean()
    #     ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
    # # 添加图例
    # ax2.legend(loc='upper right', fontsize=14)

    plt.tight_layout()
    plt.savefig("plot/长序列半衰期分布exp7.png")
    plt.show()


if __name__ == "__main__":
    pre_halflive_visualize()
    # difficulty_visualize1()
    # forgetting_curve_visualize()
    # raw_data_visualize()
    # dhp_model_visualize()
    # gru_model_visualize()
    # dhp_policy_action_visualize()
    # gru_policy_action_visualize()


def difficulty_visualize1():
    # 读取数据集
    raw = pd.read_csv('../data/opensource_dataset_difficulty.tsv', sep='\t')
    u = raw['p_recall'].mean()  # 计算召回概率的均值和标准差
    std = raw['p_recall'].std()
    print(u, std)

    plt.figure(figsize=(20, 6))  # 绘制召回概率的直方图
    ax = plt.subplot(121)
    ax.hist(raw['p_recall'],
            bins=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0],
            color='skyblue', edgecolor='black', density=False, align='mid')
    ax.set_xlabel('回忆概率（p）\n\n(a)', fontsize=28)
    ax.set_ylabel('频数', fontsize=28)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_xlim(0.15, 1)
    ax.spines[['top', 'right']].set_visible(False)  # 不显示上方和右方的坐标轴框线
    ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    ax.set_axisbelow(True)  # 网格显现在图形下方
    # plt.tight_layout()
    # plt.savefig("plot/回忆概率分布.png")
    # plt.show()

    # 绘制难度的直方图
    # plt.figure(figsize=(20, 6))
    ax = plt.subplot(122)
    # 设置背景颜色
    # ax.set_facecolor('#FFFFF0')
    ax.hist(raw['d'], bins=[0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5], color='skyblue', edgecolor='black',
            linewidth=2)
    ax.set_xlabel('难度\n\n(b)', fontsize=28)
    ax.set_ylabel('频数', fontsize=28)
    ax.set_xlim(0.5, 11)
    ax.set_xticks(range(1, 11))
    ax.tick_params(axis='x', labelsize=22)
    ax.tick_params(axis='y', labelsize=22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
    ax.set_axisbelow(False)  # 网格显现在图形下方

    plt.tight_layout()
    plt.savefig("plot/图一.png")
    plt.show()


def difficulty_visualize():
    raw = pd.read_csv('../data/opensource_dataset_difficulty.tsv', sep='\t')
    u = raw['p_recall'].mean()
    std = raw['p_recall'].std()
    print(u, std)

    fig = px.histogram(raw, x="p_recall", nbins=20)
    fig.update_xaxes(title_text='回忆概率P', title_font=dict(size=26), tickfont=dict(size=22),
                     range=[0.15, 1])
    fig.update_yaxes(title_text='数量', title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_layout(bargap=0.2, margin_t=10, margin_r=10, margin_b=10)
    fig.write_image("plot/distribution_p.png", width=600, height=360)
    # fig.show()
    time.sleep(3)

    fig = px.histogram(raw, x="d", text_auto=False)
    fig.update_xaxes(title_text='难度', title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_yaxes(title_text='数量', title_font=dict(size=26), tickfont=dict(size=22))
    fig.update_layout(bargap=0.2, margin_t=10, margin_r=10, margin_b=10)
    fig.write_image("plot/distribution_d.png", width=600, height=360)
    # fig.show()
    time.sleep(3)


def forgetting_curve_visualize():
    raw = pd.read_csv('../data/opensource_dataset_p_history.tsv', sep='\t')
    filters = [(3, '0,1', '0,1'), (3, '0,1,1', '0,1,3'), (3, '0,1,1', '0,1,4'), (3, '0,1,1', '0,1,5')]
    # filters = [(3, '0,1,1', '0,1,3')]
    fig = go.Figure()
    color = ['blue', 'red', 'green', 'orange']
    for i, f in enumerate(filters):
        print(i)
        d = f[0]
        r_history = f[1]
        t_history = f[2]
        tmp = raw[(raw['d'] == d) & (raw['r_history'] == r_history) & (raw['t_history'] == t_history)].copy()
        tmp.sort_values(by=['delta_t'], inplace=True)
        tmp['size'] = np.log(tmp['total_cnt'])  # 标点大小
        halflife = tmp['halflife'].values[0]
        tmp['fit_p_recall'] = np.power(2, -tmp['delta_t'] / halflife)  # 计算回忆概率
        fig.add_trace(
            go.Scatter(x=tmp['delta_t'], y=tmp['fit_p_recall'], mode='lines', name=f'halflife={halflife:.2f}'))
        fig.add_trace(
            go.Scatter(x=tmp['delta_t'], y=tmp['p_recall'], mode='markers', marker_size=tmp['size'],
                       name=f"d={d}|r_{{1:i-1}}={r_history}|Δt_{{1:i-1}}={t_history}"))
        fig.update_traces(marker_color=color[i], selector=dict(name=f'halflife={halflife:.2f}'))
        fig.update_traces(marker_color=color[i],
                          selector=dict(name=f"d={d}|r_{{1:i-1}}={r_history}|Δt_{{1:i-1}}={t_history}"))
    fig.update_layout(legend=dict(
        yanchor="bottom",
        y=0.01,
        xanchor="right",
        x=0.99
    ))
    fig.update_layout(yaxis=dict(range=[0.1, 1]))
    fig.update_xaxes(title_text='delta_t', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_yaxes(title_text='p_recall', title_font=dict(size=18), tickfont=dict(size=14))
    fig.update_layout(margin_t=10, margin_r=10, margin_b=10)
    # fig.show()
    fig.write_image(f"plot/forgetting_curve.png", width=1000, height=800)


def raw_data_visualize():
    raw = pd.read_csv('../data/opensource_dataset_p_history.tsv', sep='\t')
    raw.dropna(inplace=True)
    raw = raw[raw['group_cnt'] > 1000]
    raw['label'] = raw['r_history'] + '/' + raw['t_history']

    fig = px.scatter_3d(raw, x='last_p_recall', y='last_halflife',
                        z='halflife', color='d',
                        hover_name='label')
    fig.layout.scene.xaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    h_array = np.arange(0.5, 1600, 5)  # 03
    p_array = np.arange(0.3, 0.97, 0.05)  # 03
    h_array, p_array = np.meshgrid(h_array, p_array)
    fig.add_surface(y=h_array, x=p_array, z=h_array, showscale=False)
    fig.update_traces(opacity=0.2, selector=dict(type='surface'))
    fig.layout.scene.yaxis.type = 'log'
    fig.layout.scene.zaxis.type = 'log'
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_scenes(xaxis_autorange="reversed")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    print(1)
    fig.write_image(f"plot/DHP_model_raw.pdf", width=1000, height=1000)
    fig.show()


def dhp_model_visualize():
    model = DHP_HLR()
    h_array = np.arange(0.5, 750.5, 1)  # 03
    p_array = np.arange(0.97, 0.3, -0.01)  # 03
    h_array, p_array = np.meshgrid(h_array, p_array)
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_recall_halflife(diff, h_array, p_array),
                   showscale=True if diff == 10 else False
                   , cmin=0, cmax=6500
                   )
        for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_recall_model.html")
    fig.write_image(f"./plot/DHP_recall_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_recall_halflife(diff, h_array, p_array) / h_array
                   , showscale=True if diff == 10 else False
                   , cmin=0, cmax=25
                   ) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>/h<sub>i-1</sub>"
    # fig.write_html(f"./plot/DHP_recall_inc_model.html")
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    fig.write_image(f"./plot/DHP_recall_inc_model.pdf", width=1000, height=1000)
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=3), ))
    fig.update_traces(
        cmin=0,
        cmax=1.45,
        colorbar=dict(
            tickvals=[i for i in np.arange(0, 1.45, 0.2)],
            ticktext=[round(np.power(10, i)) for i in np.arange(0, 1.45, 0.2)]
        )
    )
    fig.write_image(f"./plot/DHP_recall_inc_log_model.pdf", width=1000, height=1000)
    # fig.show()
    surface = [
        go.Surface(x=h_array, y=p_array, z=model.cal_forget_halflife(diff, h_array, p_array)
                   , showscale=True if diff == 10 else False
                   , cmin=0, cmax=16
                   ) for diff in
        range(1, 11)]
    fig = go.Figure(data=surface)
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_traces(colorbar_title_text='', colorbar_tickfont_size=24, selector=dict(type='surface'))
    # fig.write_html(f"./plot/DHP_forget_model.html")
    fig.write_image(f"./plot/DHP_forget_model.pdf", width=1000, height=1000)
    # fig.show()


def gru_model_visualize():
    model = GRU_HLR()
    # for name, param in my_model.named_parameters():
    #     print(name, param)
    recall_record = np.array([0, 0, 0])
    forget_record = np.array([0, 0, 0])
    for s1, s2 in tqdm.tqdm([[x0, y0] for x0 in np.arange(-1, 1, 0.02) for y0 in np.arange(-1, 1, 0.02)]):
        h = model.state2halflife(np.array([s1, s2]))
        # # print(f'current state: {s1:.2f} {s2:.2f}\thalflife: {h: .2f}')
        # for t in np.arange(1, max(1, round(2 * h)) + 1, max(1, round(h / 10))):
        #     p = np.exp2(-t / h)
        #     n_state, nh = my_model.next_state(np.array([s1, s2]), 1, t, p)
        #     ns1, ns2 = n_state
        for p in np.arange(0.35, 0.96, 0.05):
            t = int(np.round(- np.log2(p) * h))
            p = np.exp2(-t / h)
            if t < 1 or p < 0.35:
                continue
            n_state, nh = model.next_state(np.array([s1, s2]), 1, t, p)
            # print(f'delta_t: {t}\tp_recall: {p: .3f}\tnext state: {ns1:.2f} {ns2:.2f}\thalflife: {nh: .2f}')
            recall_record = np.vstack((recall_record, np.array([h, p, nh])))
            n_state, nh = model.next_state(np.array([s1, s2]), 0, t, p)
            forget_record = np.vstack((forget_record, np.array([h, p, nh])))
    # print(record[1:, :])
    recall_model = pd.DataFrame(data=recall_record[1:, :], columns=['last_halflife', 'last_p_recall', 'halflife'])
    recall_model.drop_duplicates(inplace=True)
    recall_model['halflife_increase'] = recall_model['halflife'] / recall_model['last_halflife']
    recall_model['halflife_increase_log'] = recall_model['halflife_increase'].map(np.log)
    # recall_model['last_p_recall'] = recall_model['last_p_recall'].map(lambda x: np.round(x, decimals=2))
    # recall_model['last_halflife'] = recall_model['last_halflife'].map(lambda x: np.round(x, decimals=2))
    # last_halflife = recall_model['last_halflife'].drop_duplicates().values
    # last_p_recall = recall_model['last_p_recall'].drop_duplicates().values
    # last_halflife, last_p_recall = np.meshgrid(last_halflife, last_p_recall)
    # halflife = np.empty(last_halflife.shape)
    # halflife[:] = np.nan
    # for i in range(last_halflife.shape[0]):
    #     for j in range(last_halflife.shape[1]):
    #         halflife[i, j] = recall_model[(recall_model['last_halflife'] == last_halflife[i, j]) & (
    #                 recall_model['last_p_recall'] == last_p_recall[i, j])]['halflife'].mean()
    #
    # fig = go.Figure(data=[go.Surface(z=halflife, x=last_halflife, y=last_p_recall)])
    # fig.show()
    # exit()

    fig = px.scatter_3d(recall_model, x='last_halflife', y='last_p_recall', z='halflife', color='halflife')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_recall_model.pdf', width=1000, height=1000)
    # fig.show()

    fig = px.scatter_3d(recall_model, x='last_halflife', y='last_p_recall', z='halflife_increase',
                        color='halflife_increase_log')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>/h<sub>i-1</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_coloraxes(colorbar_tickmode='array', colorbar_tickvals=[i for i in np.arange(0, 3.5, 0.5)],
                         colorbar_ticktext=[round(np.exp(i), 1) for i in np.arange(0, 3.5, 0.5)])
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_recall_inc_model.pdf', width=1000, height=1000)
    # fig.show()
    fig.layout.scene.zaxis.type = 'log'
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=3), ))
    fig.write_image('./plot/GRU_recall_inc_log_model.pdf', width=1000, height=1000)
    # fig.show()

    forget_model = pd.DataFrame(data=forget_record[1:, :], columns=['last_halflife', 'last_p_recall', 'halflife'])
    forget_model.drop_duplicates(inplace=True)

    fig = px.scatter_3d(forget_model, x='last_halflife', y='last_p_recall', z='halflife', color='halflife')
    fig.layout.scene.xaxis.title = r"h<sub>i-1</sub>"
    fig.layout.scene.yaxis.title = r"p<sub>i-1</sub>"
    fig.layout.scene.zaxis.title = r"h<sub>i</sub>"
    fig.update_traces(marker_size=2, selector=dict(type='scatter3d'))
    fig.update_layout(
        scene_camera=camera,
        scene=dict(
            xaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)),
            yaxis=dict(title_font=dict(size=24), tickfont=dict(size=16), nticks=6),
            zaxis=dict(title_font=dict(size=24), tickfont=dict(size=16)), ))
    fig.update_layout(margin_b=50, margin_t=50, margin_l=0, margin_r=50, margin_pad=100)
    fig.update_coloraxes(colorbar_tickfont_size=24)
    fig.update_coloraxes(colorbar_title_text='')
    fig.write_image('./plot/GRU_forget_model.pdf', width=1000, height=1000)
    # fig.show()

# def pre_halflive_visualize():
#     # 绘制半衰期的分布
#     plt.figure(figsize=(10, 10))  # 绘制召回概率的直方图
#     ax = plt.subplot(111)
#     raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
#     counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     ax.set_xlabel('半衰期（h）\n(a)', fontsize=28)
#     ax.set_ylabel('样本数量\n', fontsize=28)
#     ax.tick_params(axis='both', labelsize=22)
#     ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
#     ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
#                   ['1', '3', '10', '30', '100', '1000'])
#     ax.set_axisbelow(True)  # 网格显现在图形下方
#     # 绘制不同sMAPE的误差分布
#     ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
#     ax2.set_ylabel('\nsMAPE', fontsize=28)
#     ax2.tick_params(axis='y', labelsize=22)
#     ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     files = [
#         'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
#         'static-Transformer-d_model=256-nhead=4-encoder_num=4',
#         '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
#         '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',
#         'DHP', 'HLR', 'HLR-lex']
#     colors = [
#         'green', 'purple',
#         'orange', 'yellow', 'brown', 'cyan',
#         'pink', 'magenta', 'lime']
#     labels = [
#         'Transformer-HLR','Transformer-HLR+d',
#         'GRU-HLR', 'GRU&Attention-HLR',
#         'GRU-HLR-t', 'GRU&Attention-HLR -t',
#         'DHP-HLR', 'HLR', 'HLR-lex']
#     for filename, color, label in zip(files, colors, labels):
#         raw = concat_data(filename)
#         raw['bin'] = pd.cut(raw['halflife'], bins=bins)
#         bin_means = raw.groupby('bin', observed=True)['sape'].mean()
#         ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
#     # 添加图例
#     ax2.legend(loc='upper right', fontsize=14)
#
#     # ax = plt.subplot(132)
#     # raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
#     # counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
#     # bin_centers = (bins[:-1] + bins[1:]) / 2
#     # ax.set_xlabel('半衰期（h）\n(b)', fontsize=28)
#     # ax.set_ylabel('样本数量\n', fontsize=28)
#     # ax.tick_params(axis='both', labelsize=22)
#     # ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
#     # ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     # ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
#     #               ['1', '3', '10', '30', '100', '1000'])
#     # ax.set_axisbelow(True)  # 网格显现在图形下方
#     # # 绘制不同sMAPE的误差分布
#     # ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
#     # ax2.set_ylabel('\nsMAPE', fontsize=28)
#     # ax2.tick_params(axis='y', labelsize=22)
#     # ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     # files = [
#     #     'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
#     #     'static-Transformer-d_model=256-nhead=4-encoder_num=4',]
#     #     # '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
#     #     # '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',
#     #     # 'DHP', 'HLR', 'HLR-lex']
#     # colors = [
#     #     'green', 'purple',]
#     #     # 'orange', 'yellow', 'brown', 'cyan',
#     #     # 'pink', 'magenta', 'lime']
#     # labels = [
#     #     'Transformer-HLR', 'Transformer-HLR+d',]
#     #     # 'GRU-HLR', 'GRU&Attention-HLR',
#     #     # 'GRU-HLR-t', 'GRU&Attention-HLR -t',
#     #     # 'DHP-HLR', 'HLR', 'HLR-lex']
#     # for filename, color, label in zip(files, colors, labels):
#     #     raw = concat_data(filename)
#     #     raw['bin'] = pd.cut(raw['halflife'], bins=bins)
#     #     bin_means = raw.groupby('bin', observed=True)['sape'].mean()
#     #     ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
#     # # 添加图例
#     # ax2.legend(loc='upper right', fontsize=14)
#     #
#     # ax = plt.subplot(133)
#     # raw = concat_data('exp-Transformer-d_model=256-nhead=4-encoder_num=4')
#     # counts, bins, _ = ax.hist(raw['halflife'], bins=40, color='skyblue', edgecolor='black', density=False, align='mid')
#     # bin_centers = (bins[:-1] + bins[1:]) / 2
#     # ax.set_xlabel('半衰期（h）\n(c)', fontsize=28)
#     # ax.set_ylabel('样本数量\n', fontsize=28)
#     # ax.tick_params(axis='both', labelsize=22)
#     # ax.spines[['top', 'right']].set_visible(True)  # 上方和右方的坐标轴框线
#     # ax.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     # ax.set_xticks([0, math.log(3), math.log(10), math.log(30), math.log(100), math.log(1000)],
#     #               ['1', '3', '10', '30', '100', '1000'])
#     # ax.set_axisbelow(True)  # 网格显现在图形下方
#     # # 绘制不同sMAPE的误差分布
#     # ax2 = ax.twinx()  # 在 y 轴右边创建一个新的轴对象
#     # ax2.set_ylabel('\nsMAPE', fontsize=28)
#     # ax2.tick_params(axis='y', labelsize=22)
#     # ax2.yaxis.grid(linewidth=0.5, color="grey", alpha=0.5)
#     # files = [
#     #     # 'exp-Transformer-d_model=256-nhead=4-encoder_num=4',
#     #     # 'static-Transformer-d_model=256-nhead=4-encoder_num=4', ]
#     # '2nn-GRU_nh-2_loss-sMAPE', '2nn-GRU_nh-2_loss-sMAPE-atten',
#     # '2nn-GRU_nh-2_loss-sMAPE-t', '2nn-GRU_nh-2_loss-sMAPE-t-atten',]
#     # # 'DHP', 'HLR', 'HLR-lex']
#     # colors = [
#     #     # 'green', 'purple', ]
#     # 'orange', 'yellow', 'brown', 'cyan',]
#     # # 'pink', 'magenta', 'lime']
#     # labels = [
#     #     # 'Transformer-HLR', 'Transformer-HLR+d', ]
#     # 'GRU-HLR', 'GRU&Attention-HLR',
#     # 'GRU-HLR-t', 'GRU&Attention-HLR -t',]
#     # # 'DHP-HLR', 'HLR', 'HLR-lex']
#     # for filename, color, label in zip(files, colors, labels):
#     #     raw = concat_data(filename)
#     #     raw['bin'] = pd.cut(raw['halflife'], bins=bins)
#     #     bin_means = raw.groupby('bin', observed=True)['sape'].mean()
#     #     ax2.plot(bin_centers, bin_means, color=color, linestyle='-', marker='', markersize=5, label=label)
#     # # 添加图例
#     # ax2.legend(loc='upper right', fontsize=14)
#
#     plt.tight_layout()
#     plt.savefig("plot/半衰期分布exp2.png")
#     plt.show()
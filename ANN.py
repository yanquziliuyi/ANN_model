from sklearn.neural_network import MLPRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator,MultipleLocator,FuncFormatter
import pickle

train = r"train_data_400simple.xlsx"
test = r"test_data.xlsx"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
for moniwell in range(6):
    # 读取训练数据：输入为10个应力期注水量，输出为9个观测井浓度，总计500个train set
    X_dataset = pd.read_excel(train, sheet_name=0)
    y_dataset = pd.read_excel(train, sheet_name=1)
    X = X_dataset.iloc[:, 1:].values
    y_train = y_dataset.iloc[:, 1:].values
    X_train = X
    row_train = X.shape[0]
    y_train = y_train[:row_train, moniwell]
    y_train = np.reshape(y_train, (row_train,))

    ANN = MLPRegressor(max_iter=10000, random_state=0,
                       alpha=0.01, solver='lbfgs',
                       activation='logistic', hidden_layer_sizes=(12,)).fit(X_train, y_train)

    X1_dataset = pd.read_excel(test, sheet_name=0)
    y1_dataset = pd.read_excel(test, sheet_name=1)
    X1 = X1_dataset.iloc[:, 1:].values
    y_test = y1_dataset.iloc[:, 1:].values
    X_test = X1
    row_test = X_test.shape[0]
    y_test = y_test[:row_test, moniwell]
    y_test = np.reshape(y_test, (row_test,))

    y_pred = ANN.predict(X_test)

    y_pred[y_pred>35] = 35


    with open(r'ANN_pickle\ANN预测观测点{}.pickle'.format(moniwell + 1), 'wb') as f:
        pickle.dump(ANN, f)

    from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

    r_squared = r2_score(y_test, y_pred)
    r_squared = "{:.4}".format(r_squared)
    MAPE = mean_absolute_percentage_error(y_test, y_pred)
    MAPE = "{:.4}".format(MAPE)
    RMSE = mean_squared_error(y_test, y_pred)
    RMSE = "{:.4}".format(RMSE)

    plt.rc('font', family='Times New Roman')
    fig = plt.figure(num=1, figsize=(8, 8), dpi=500)
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    pointsize=30
    linewidth=2
    ax.scatter(y_test, y_pred, color='k', s=pointsize)
    ax.plot((0, 1), (0, 1), transform=ax.transAxes, color='k',linewidth=linewidth)
    if moniwell == 0:
        low = 3.5
        up = 11.5
    if moniwell == 1:
        low = 2.5
        up = 17.5
    if moniwell == 2:
        low = 1.5
        up = 20.5
    if moniwell == 3:
        low = 2.5
        up = 22.5
    if moniwell == 4:
        low = 4
        up = 33.0
    if moniwell == 5:
        low = 0
        up = 35.5

    my_x_ticks = np.linspace(low, up, num=6, endpoint=True)
    my_x_ticks = np.around(my_x_ticks, decimals=1)

    ax.spines['top'].set_linewidth('2.0')
    ax.spines['right'].set_linewidth('2.0')
    ax.spines['bottom'].set_linewidth('2.0')
    ax.spines['left'].set_linewidth('2.0')

    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 33}

    plt.figtext(0.1, 0.90, f'NSE = {r_squared}', transform=ax.transAxes, fontdict=font3)
    plt.figtext(0.1, 0.80, f'MAPE = {MAPE}', transform=ax.transAxes, fontdict=font3)
    plt.figtext(0.1, 0.70, f'RMSE = {RMSE}', transform=ax.transAxes, fontdict=font3)
    plt.figtext(0.75, 0.20, 'ANN', transform=ax.transAxes, fontdict=font3)
    ax.set_xticks(my_x_ticks, )
    ax.set_yticks(my_x_ticks, )

    ax.tick_params(axis='both',
                   which='major',
                   labelsize=30,  # y轴字体大小设置
                   color='k',  # y轴标签颜色设置
                   labelcolor='k',  # y轴字体颜色设置
                   direction='in',  # y轴标签方向设置
                   pad=11,
                   tick1On=True,
                   tick2On=True,
                   label1On=True,
                   length=12,
                   width=1.5, )

    ax.tick_params(axis='both',
                   which="minor",
                   labelsize=10,
                   color='k',
                   direction='in',
                   tick1On=True,
                   tick2On=True,
                   length=6,
                   width=1.0, )

    ax.set(xlim=(low, up), ylim=(low, up))

    ax.set_xlabel('Simulated C (g/L)', font3)
    ax.set_ylabel('Surrogate C (g/L)', font3)

    # plt.rcParams['xtick.direction'] = 'in'
    # plt.rcParams['ytick.direction'] = 'in'

    ax.xaxis.set_minor_locator(AutoMinorLocator(3))  # 设置x轴次刻度
    ax.yaxis.set_minor_locator(AutoMinorLocator(3))  # 设置y轴次刻度

    plt.savefig(r"ANN_fitting_figure\Observation_point{}.eps".format(moniwell + 1), bbox_inches='tight')
    plt.close()

    if moniwell == 0:
        X_dataset = pd.read_excel(train, sheet_name=0)
        y_dataset = pd.read_excel(train, sheet_name=2)
        X = X_dataset.iloc[:, 1:].values
        y_train = y_dataset.iloc[:, 1:].values
        X_train = X
        row_train = X.shape[0]
        y_train = y_train[:row_train, moniwell]
        y_train = np.reshape(y_train, (row_train,))
        ANN = MLPRegressor(max_iter=10000, random_state=0,
                           alpha=0.01, solver='lbfgs',
                           activation='tanh', hidden_layer_sizes=(12,)).fit(X_train, y_train)
        with open(r'ANN_pickle\ANN预测质量.pickle', 'wb') as f:
            pickle.dump(ANN, f)

        X1_dataset = pd.read_excel(test, sheet_name=0)
        y1_dataset = pd.read_excel(test, sheet_name=2)
        X1 = X1_dataset.iloc[:, 1:].values
        y_test = y1_dataset.iloc[:, 1:].values
        X_test = X1
        row_test = X_test.shape[0]
        y_test = y_test[:row_test, moniwell]
        y_test = np.reshape(y_test, (row_test,))
        y_pred = ANN.predict(X_test)
        from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error

        r_squared = r2_score(y_test, y_pred)
        r_squared = "{:.4}".format(r_squared)
        MAPE = mean_absolute_percentage_error(y_test, y_pred)
        MAPE = "{:.4}".format(MAPE)
        RMSE = mean_squared_error(y_test, y_pred)
        RMSE = "{:.4}".format(RMSE)

        plt.rc('font', family='Times New Roman')
        fig = plt.figure(num=1, figsize=(8, 8), dpi=500)

        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        ax.spines['top'].set_linewidth('2.0')
        ax.spines['right'].set_linewidth('2.0')
        ax.spines['bottom'].set_linewidth('2.0')
        ax.spines['left'].set_linewidth('2.0')

        ax.scatter(y_test, y_pred, color='k', s=pointsize)
        ax.plot((0, 1), (0, 1), transform=ax.transAxes, color='k',linewidth=linewidth)

        low = 1.5
        up = 2.5
        my_x_ticks = np.linspace(low, up, num=6, endpoint=True)
        my_x_ticks = np.around(my_x_ticks, decimals=1)

        plt.figtext(0.1, 0.90, f'NSE = {r_squared}',  transform=ax.transAxes, fontdict=font3 )
        plt.figtext(0.1, 0.80, f'MAPE = {MAPE}',  transform=ax.transAxes,  fontdict=font3)
        plt.figtext(0.1, 0.70, f'RMSE = {RMSE}',  transform=ax.transAxes,  fontdict=font3)
        plt.figtext(0.75, 0.20, 'ANN',  transform=ax.transAxes,  fontdict=font3)
        ax.set_xticks(my_x_ticks, )
        ax.set_yticks(my_x_ticks, )
        ax.tick_params(axis='both',
                       which='major',
                       labelsize=30,  # y轴字体大小设置
                       color='k',  # y轴标签颜色设置
                       labelcolor='k',  # y轴字体颜色设置
                       direction='in',  # y轴标签方向设置
                       pad=11,
                       tick1On=True,
                       tick2On=True,
                       label1On=True,
                       length=12,
                       width=1.5,)

        ax.tick_params(axis='both',
                       which="minor",
                       labelsize=10,
                       color='k',
                       direction='in',
                       tick1On=True,
                       tick2On=True,
                       length=6,
                       width=1.0,)

        plt.figtext(0.1, 0.90, f'NSE = {r_squared}', transform=ax.transAxes, fontdict=font3)
        plt.figtext(0.1, 0.80, f'MAPE = {MAPE}', transform=ax.transAxes, fontdict=font3)
        plt.figtext(0.1, 0.70, f'RMSE = {RMSE}', transform=ax.transAxes, fontdict=font3)
        plt.figtext(0.75, 0.20, 'ANN', transform=ax.transAxes, fontdict=font3)
        ax.set_xticks(my_x_ticks,)
        ax.set_yticks(my_x_ticks,)

        ax.set(xlim=(min(my_x_ticks), max(my_x_ticks)), ylim=(min(my_x_ticks), max(my_x_ticks)))
        ax.set_xlabel('Simulated M (g)', font3,)
        ax.set_ylabel('Surrogate M (g)', font3,)

        ax.xaxis.set_minor_locator(AutoMinorLocator(3)) # 设置x轴次刻度
        ax.yaxis.set_minor_locator(AutoMinorLocator(3)) # 设置y轴次刻度

        plt.savefig(r"ANN_fitting_figure\ANN_预测质量.eps", bbox_inches='tight')
        plt.close()

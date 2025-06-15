# import pandas as pd
#
# pd.set_option('display.max_columns', None)    # 显示所有列
# pd.set_option('display.max_rows', None)      # 显示所有行
# pd.set_option('max_colwidth', 200)
#
# path = "E:\\2023C题\\附件3.xlsx"
# excel3 = pd.read_excel(path)
# print(excel3[0:100])
# print('---------------------------------------------')
# #print(len(pd.DataFrame(excel3).groupby('单品编码')))
# print('---------------------------------------------')
# print(*pd.DataFrame(excel3[0:100]).groupby('单品编码'))
# print('---------------------------------------------')
# print(*pd.DataFrame(excel3[0:100]).groupby('单品编码')['0'])


import numpy as np
from sklearn.linear_model import LinearRegression

# 股票收盘价数据（10天）
closing_prices = np.array([100, 102, 105, 107, 110, 108, 107, 111, 114, 115])

# 定义滑动窗口的大小
window_size = 5  # 使用5天的收盘价作为训练数据

# 初始化线性回归模型
model = LinearRegression()

# 用来存储实际价格和预测价格，以便之后分析
actual_prices = []
predicted_prices = []

# 模拟滑动窗口交叉验证
for i in range(len(closing_prices) - window_size):
    train_data = closing_prices[i:i + window_size]  # 训练集为当前窗口内的数据
    test_data = closing_prices[i + window_size]  # 测试集为窗口之后的一个数据点

    # 构造训练集的特征和目标变量
    X_train = np.arange(window_size).reshape(-1, 1)  # X_train: [0, 1, 2, 3, 4]
    y_train = train_data  # y_train: 对应的股价

    # 训练线性回归模型
    model.fit(X_train, y_train)

    # 使用模型预测测试集的值
    X_test = np.array([[window_size]])  # 下一个时间点
    predicted_price = model.predict(X_test)[0]  # 预测股价

    # 保存实际价格和预测价格
    actual_prices.append(test_data)
    predicted_prices.append(predicted_price)

    # 输出当前的训练集、实际测试集股价和预测的股价
    print(f"训练集: {train_data}, 实际股价: {test_data}, 预测股价: {predicted_price:.2f}")

# 这里你可以进一步计算预测误差，如均方误差（MSE）、平均绝对误差（MAE）等。

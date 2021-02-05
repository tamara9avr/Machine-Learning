import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. READ DATA

data = pd.read_csv("house_prices_train.csv")
print(data.head())
print(data.tail())

# 2. DATA ANALYSIS
print(data.info())
print(data.describe())

# 3. PLOT

labels = data.loc[:, 'Price']

features = ['Year_built', 'Area', 'Bath_no', 'Bedroom_no']

fig = plt.figure('Prices', figsize=(8, 5), dpi=190)

for j in range(len(features)):
    ax1 = fig.add_subplot(221 + j)
    x = data.loc[:, features[j]]
    ax1.scatter(x, labels, s=10, c='red', marker='o', alpha=0.7, edgecolors='black', linewidths=1, label='houses')
    ax1.set_xlabel(features[j], fontsize=13)
    ax1.set_ylabel('Price', fontsize=13)
    ax1.set_title('Price for ' + features[j])
    ax1.legend()

plt.tight_layout()
# plt.show()
fig.savefig('features.png')


# 4. DATA TRANSFORMATION
# Vrsi se normalizacija podataka da bi svi bili u istom opsegu, da bi podjednako uticali na resenje


def normalize(data_input):
    mincol = np.min(data_input, axis=0)
    maxcol = np.max(data_input, axis=0)
    data_norm = (data_input - mincol) / (maxcol - mincol)
    return data_norm


data = normalize(data)
print(data.head())

# 5. FEATURE ENGINEERING
data_train = data.loc[:, features]


# 6. LINEAR REGRESSION GRADIENT DESCENT

class GradientDescent:
    def __init__(self):
        self.coeff = None
        self.features = None
        self.target = None
        self.mse_history = None

    def set_coefficients(self, *args):
        self.coeff = np.array(args).reshape(-1, 1)

    def cost(self):
        predicted = self.features.dot(self.coeff)
        s = pow(predicted - self.target, 2).sum()
        return (0.5 / len(self.features)) * s

    def predict(self, features):
        features = features.copy(deep=True)
        features.insert(0, 'c0', np.ones((len(features), 1)))
        features = features.to_numpy()
        return features.dot(self.coeff).reshape(-1, 1).flatten()

    def gradient_descent_step(self, learning_rate=0.01):
        predicted = self.features.dot(self.coeff)
        s = self.features.T.dot(predicted - self.target)
        gradient = (1. / len(self.features)) * s
        self.coeff = self.coeff - learning_rate * gradient
        return self.coeff, self.cost()

    def perform_gradient_descent(self, learning_rate=0.5, num_iterations=1300):
        self.mse_history = []
        for i in range(num_iterations):
            _, curr_cost = self.gradient_descent_step(learning_rate)
        self.mse_history.append(curr_cost)
        return self.coeff, self.mse_history

    def fit(self, features, target):
        self.features = features.copy(deep=True)
        coeff_shape = len(features.columns) + 1
        self.coeff = np.zeros(shape=coeff_shape).reshape(-1, 1)
        self.features.insert(0, 'c0', np.ones((len(features), 1)))
        self.features = self.features.to_numpy()
        self.target = target.to_numpy().reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.8, random_state=345, shuffle=True)


lrgd = GradientDescent()
lrgd.fit(x_train, y_train)
res_coeff, mse_history = lrgd.perform_gradient_descent()
price_predicted_gd = lrgd.predict(x_test)
ser_pred_gd = pd.Series(data=price_predicted_gd, name='Predicted', index=x_test.index)
res_df_gd = pd.concat([x_test, y_test, ser_pred_gd], axis=1)

print(res_df_gd.head())

# 7. LINEAR REGRESSION

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
price_predicted = lr_model.predict(x_test)
ser_pred = pd.Series(data=price_predicted, name='Predicted', index=x_test.index)
res_df = pd.concat([x_test, y_test, ser_pred], axis=1)

print(res_df.head())


lr_coef_ = lr_model.coef_
lr_int_ = lr_model.intercept_
helparr = lrgd.coeff[1:]
helparr = helparr.reshape(-1)
lr_model.coef_ = helparr
lr_model.intercept_ = lrgd.coeff[0][0]
print(f'LRGD score: {lr_model.score(x_test, y_test):.2f}')
lr_model.coef_ = lr_coef_
lr_model.intercept_ = lr_int_
print(f'LR score: {lr_model.score(x_test, y_test):.2f}')


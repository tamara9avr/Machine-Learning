import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sb

# 1.DATA READING

pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
data = pd.read_csv('cakes_train.csv')
print(data.head())
print(data.tail())

# 2.DATA ANALYSIS

print(data.info())
print(data.describe())

# 4.DATA TRANSFORMATION

features = ['flour', 'eggs', 'sugar', 'milk', 'butter', 'baking_powder']

labels = data.loc[:, 'type']

data_train = data.loc[:, features]
eggs = data_train.eggs

# Kolicina jaja se prebacuje u grame da bi bila u istoj jedinici kao ostali atributi
eggs = [e * 63 for e in eggs]

data_train.drop(columns='eggs', inplace=True)
data_train = data_train.join(pd.DataFrame(data=eggs, columns=['eggs']))

# Racuna se procenat svakog sastojka u receptu zato sto nisu jednake kolicine kolaca

data_train = data_train.div(data_train.sum(axis=1), axis=0)*100

# 3.PLOTTING

fig = plt.figure('Types of cakes', figsize=(10, 5), dpi=180)

for i in range(len(features)):
    ax1 = fig.add_subplot(231+i)
    x = data_train.loc[:, features[i]]
    ax1.scatter(x, labels, s=20, c='brown', marker='o', alpha=0.7, edgecolors='black', linewidths=1)
    ax1.set_xlabel(features[i], fontsize=10)
    ax1.set_ylabel('Type', fontsize=10)
    ax1.set_title('Ingredient ' + features[i])

plt.tight_layout()
fig.savefig('types_of_cake.png')

fig1 = plt.figure('Heatmap', figsize=(10, 8), dpi=500)
sb.heatmap(data.corr(), annot=True, fmt='.2f')
fig1.savefig('Heatmap.png')



# 5. MODEL TRAINING

dtc_model = DecisionTreeClassifier(criterion='entropy')
x_train, x_test, y_train, y_test = train_test_split(data_train, labels, train_size=0.8, random_state=234, shuffle=True)
dtc_model.fit(x_train, y_train)
labels_predicted = dtc_model.predict(x_test)
ser_pred = pd.Series(data=labels_predicted, name='Predicted', index=x_test.index)
res_df = pd.concat([x_test, y_test, ser_pred], axis=1)

print(res_df.head())
print(dtc_model.score(x_test, y_test))

fig, axis = plt.subplots(1, 1, figsize=(10, 8), dpi=300)
tree.plot_tree(decision_tree=dtc_model, max_depth=6, feature_names=data_train.columns,
               class_names=['muffin', 'cupcake'], fontsize=10, filled=True)

fig.savefig('tree.png')


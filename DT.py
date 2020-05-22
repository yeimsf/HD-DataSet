import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
df = pd.read_csv("heart.csv")
# IN [2]: to see if dataset is read successfuly
df.head(10)
df.info()
# IN [3]: to describe columns and output their mode median mean std quartiles , etc.
df['age'].describe()
df['trestbps'].describe()
df['chol'].describe()
df['thalach'].describe()
df['oldpeak'].describe()
df['thal'].describe()
df['cp'].describe()
df['sex'].descirbe()
df['fbs'].describe()
df['restecg'].describe()
df['slope'].describe()
df['ca'].describe()
df['target'].describe()
df['exang'].describe()
#Outliers Calculations for the 5 stated attributes
Q1 = df['age'].quantile(0.25)
Q3 = df['age'].quantile(0.75)
IQR = Q3 - Q1
Out1,Out2 = (Q1 - 1.5 * IQR),(Q3 + 1.5 * IQR)
print(Out1,Out2)
Q1 = df['cp'].quantile(0.25)
Q3 = df['cp'].quantile(0.75)
IQR = Q3 - Q1
Out1,Out2 = (Q1 - 1.5 * IQR),(Q3 + 1.5 * IQR)
print(Out1,Out2)
Q1 = df['chol'].quantile(0.25)
Q3 = df['chol'].quantile(0.75)
IQR = Q3 - Q1
Out1,Out2 = (Q1 - 1.5 * IQR),(Q3 + 1.5 * IQR)
print(Out1,Out2)
Q1 = df['thal'].quantile(0.25)
Q3 = df['thal'].quantile(0.75)
IQR = Q3 - Q1
Out1,Out2 = (Q1 - 1.5 * IQR),(Q3 + 1.5 * IQR)
print(Out1,Out2)
Q1 = df['thalach'].quantile(0.25)
Q3 = df['thalach'].quantile(0.75)
IQR = Q3 - Q1
Out1,Out2 = (Q1 - 1.5 * IQR),(Q3 + 1.5 * IQR)
print(Out1,Out2)

# IN [4]: to plot the boxplots and shapes of attributes.
fig=plt.figure(figsize=(15,30))

sns.set(style="whitegrid")
cp = fig.add_subplot(721)
cp = sns.boxplot(x="cp", hue="target", data=df, palette="Set3")

chol = fig.add_subplot(722)
chol = sns.boxplot(x="chol", hue="target", data=df, palette="Set3")

thal = fig.add_subplot(725)
thal = sns.boxplot(x="thal", hue="target", data=df, palette="Set3")

thalach = fig.add_subplot(726)
thalach = sns.boxplot(x="thalach", hue="target", data=df, palette="Set3")

plt.show()

fig.add_subplot(7,2,1)
sns.countplot(x=pd.cut(df['age'], bins=np.arange(20,90,10)),hue='target',data=df)

fig.add_subplot(7,2,2)
sns.countplot(x='sex',hue='target',data=df)

fig.add_subplot(7,2,3)
sns.countplot(x='cp',hue='target',data=df)

fig.add_subplot(7,2,4)
sns.countplot(x=pd.cut(df['trestbps'], bins=np.arange(90,220,30)),hue='target',data=df)

fig.add_subplot(7,2,5)
sns.countplot(x=pd.cut(df['chol'], bins=np.arange(120,580,80)),hue='target',data=df)

fig.add_subplot(7,2,6)
sns.countplot(x='fbs', hue='target',data=df)

fig.add_subplot(7,2,7)
sns.countplot(x='restecg',hue='target',data=df)

fig.add_subplot(7,2,8)
sns.countplot(x=pd.cut(df['thalach'], bins=np.arange(70,230,30)),hue='target',data=df)

fig.add_subplot(7,2,9)
sns.countplot(x='exang',hue='target',data=df)

fig.add_subplot(7,2,10)
sns.countplot(x=pd.cut(df['oldpeak'], bins=np.arange(0,8,1)),hue='target',data=df)

fig.add_subplot(7,2,11)
sns.countplot(x='slope',hue='target',data=df)

fig.add_subplot(7,2,12)
sns.countplot(x='ca',hue='target',data=df)

fig.add_subplot(7,2,13)
sns.countplot(x='thal',hue='target',data=df)

plt.show()
# IN [5]: to generate the heatmap for correlation.
plt.figure(figsize=(12,10))
sns.heatmap(abs(df.corr()), annot=True) #Shows the features correlation. ps: there's a bug on matplotlib #3.1.1.
plt.show()
# IN [6]: to split the data and balance it.
X_train, X_test, y_train, y_test = train_test_split(df.drop(['target'], axis =1), df['target'], test_size=0.2, random_state=2)
X_train.head()
X_train.shape
# checking if balanced?
y_train.mean()
sm = SMOTE(random_state=1)
X_train_balanced, y_train_balanced= sm.fit_resample(X_train, y_train)
# Balancing it
X_train_balanced.shape
y_train_balanced.shape
y_train_balanced.mean()
# IN [7]: Building models Decision Tree and Random Forest.
dt = DecisionTreeClassifier(random_state=2)
print('CV score:', cross_val_score(dt, X_train_balanced, y_train_balanced, cv = 3).mean())
rf = RandomForestClassifier(n_estimators=100, random_state=2)
print('CV score:', cross_val_score(rf, X_train_balanced, y_train_balanced, cv = 3).mean())
rf_grid = RandomForestClassifier(random_state=2) # creates a new estimator
# IN [8]: Random Forest.
# Create the parameter grid based on the results of random search
rf_param_grid = {'criterion' : ['gini', 'entropy'],
              'min_samples_leaf': [1, 2, 3],
              'min_samples_split': [2, 3, 5, 10],
              'n_estimators': [100, 300, 500]}

# Instantiate the grid search model (n_jobs = -1 sets to use the max number of processors)
rf_grid_search = GridSearchCV(estimator=rf_grid, param_grid=rf_param_grid, cv=3, scoring='precision', n_jobs=-1, verbose=2)

rf_grid_search.fit(X_train_balanced, y_train_balanced)
rf_grid_search.best_params_
# IN [9]: Exploring Results and showing the plot bar.
rf_best_grid.feature_importances_
importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(rf_best_grid.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
importances.plot.bar()
plt.show()
# IN [10]: Final Build of the tree.
export_graphviz(rf_best_grid.estimators_[0], out_file='tree.dot',
                feature_names = X_train.columns,
                class_names = ['no disease', 'disease'],
                rounded = True, proportion = True,
                label='root',
                precision = 2, filled = True)

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=300'])

Image(filename = 'tree.png')
# Tree Done.

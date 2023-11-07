#Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

#Importing the dataset
url='https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv'
df = pd.read_csv(url)

#Exploring the data
df.head()
df.info()

#Checking for duplicates
df.duplicated().sum()

#Dropping duplicates
df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

#Checking for null values
df.isnull().sum()
sns.heatmap(df.isnull())

#Descriptive statistics and plot of categorical variables
cat_variables = df.describe(include=['O'])
cat_variables

#Plots
col_list = [i for i in cat_variables.columns]

num_plots = len(col_list) + 1
total_cols = 3
total_rows = num_plots//total_cols
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,figsize=(7*total_cols, 6*total_rows), constrained_layout=True)

index = 0
for col in col_list:

    row = index //total_cols
    pos = index % total_cols
    sns.histplot(ax=axs[row][pos], data=df, x=df[col])
    axs[row][pos].set_xticklabels(axs[row][pos].get_xticklabels(), rotation=90, fontsize=16)
    axs[row][pos].set_yticklabels(axs[row][pos].get_yticklabels(), fontsize=16)
    axs[row][pos].set_xlabel(axs[row][pos].get_xlabel(),fontsize=16)
        
    index += 1
    
plt.tight_layout()
plt.show()

#Descriptive statistics and plot of numerical variables
num_variables = df.describe()
num_variables

#Plots
col_n_list = [i for i in num_variables.columns]

num_plots = len(col_n_list)
total_cols = 2
total_rows = num_plots//total_cols
fig, axs = plt.subplots(nrows=total_rows, ncols=total_cols,figsize=(6*total_cols, 3*total_rows), constrained_layout=True)

index = 0
for col in col_n_list:

    row = index //total_cols
    pos = index % total_cols
    sns.distplot(df[col], rug=False, ax=axs[row][pos])
    
    index += 1
    
plt.tight_layout()
plt.show()

#Pairplot
sns.pairplot(df)

#Creating a copy of the dataframe for feature engineering
df2 = df.copy()

#Factorizing categorical variables
for i in col_list:
    df2[i] = pd.factorize(df2[i])[0]

#Normalising numerical variables
scaler = MinMaxScaler()
df2[col_n_list] = scaler.fit_transform(df[col_n_list])

#Correlation matrix
fig = plt.figure(figsize=(6,6))
corr_matrix = df2.corr()
sns.heatmap(corr_matrix)
plt.tight_layout()
plt.show()

#Data split for machine learning model
X = df2.drop('y', axis=1)
y = df2['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train data:\nX:', X_train.shape, 'y:', y_train.shape)
print('Test data:\nX:', X_test.shape, 'y:', y_test.shape)

#Feature selection
selection_model = SelectKBest(chi2, k=10)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

#Logistic Regression model
model = LogisticRegression()
model.fit(X_train_sel, y_train)

#Model prediction
ytrain = model.predict(X_train_sel)
y_pred = model.predict(X_test_sel)

#Accuracy score
accuracy_train = accuracy_score(y_train, ytrain)
print(f'Accuracy on train data is {accuracy_train:.2f}')
accuracy_predict = accuracy_score(y_test, y_pred)
print(f'Accuracy on test data is {accuracy_predict:.2f}')

#F1 Score works better for unbalanced data
print(f'F1 Score Train: {f1_score(y_train, ytrain):.2f}')
print(f'F1 Score Test: {f1_score(y_test, y_pred):.2f}')

#Confusion matrix
bank_cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(bank_cm)
plt.figure(figsize=(2,2))
sns.heatmap(cm_df, annot=True, fmt="d", cbar=False)
plt.tight_layout()
plt.show()

#Model optimization: Grid Search
hyperparams = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

grid = GridSearchCV(model, hyperparams, scoring = "f1", cv = 5)
grid.fit(X_train_sel, y_train)

print(f"Best hyperparameters: {grid.best_params_}")

model_grid = LogisticRegression(C=1000, penalty="l1", solver="liblinear")
model_grid.fit(X_train_sel, y_train)
y_pred = model_grid.predict(X_test_sel)

print(f'After model optimization with grid search method the accuracy score of the test data is {accuracy_score(y_test, y_pred):.2f} and the F1 score is {f1_score(y_test, y_pred):.2f}.')

#Model optimization: Random Search
hyperparams = {
    "C": np.logspace(-4, 4, 20),
    "penalty": ["l1", "l2", "elasticnet", None],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}

random_search = RandomizedSearchCV(model, hyperparams, n_iter=100, scoring="f1", cv=5, random_state=42)
random_search.fit(X_train_sel, y_train)

print(f"Best hyperparameters: {random_search.best_params_}")

model_random_search = LogisticRegression(penalty = "l1", C = 10000.0, solver = "liblinear")
model_random_search.fit(X_train_sel, y_train)
y_pred = model_random_search.predict(X_test_sel)

print(f'After model optimization with random search method the accuracy score of the model is {accuracy_score(y_test, y_pred):.2f} and the F1 score is {f1_score(y_test, y_pred):.2f}')
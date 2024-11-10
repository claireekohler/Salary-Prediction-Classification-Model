import sklearn 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score, f1_score
from sklearn import svm, metrics
import seaborn as sns
from sklearn.inspection import permutation_importance
import sys
from sklearn.svm import SVC
import xgboost as xgb
from xgboost import XGBClassifier


print(sys.executable)
# read in csv
df = pd.read_csv('salary.csv')
#df = df.reset_index()
raw = df
print(raw)

#drop all na and blank entries
df = df.replace(' ?', None)
df = df.dropna(axis=0, how = 'any')

#EDA
df.info()
df.describe()

#distribution of salary among education and age
sns.pairplot(df, hue='salary', vars=['age','education-num'], kind='kde') 
plt.show()

#Count plot - sex by salary
ax = sns.countplot(x='sex', hue='salary', data=df)
plt.title('Count Plot of sex by salary')


for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, int(height), ha="center")

plt.show()

#Male dominated - more than half of the dataset

#distribution of salary among capital gain
plt.figure(figsize=(10, 6))
hist = sns.histplot(df,x='capital-gain',hue='salary',bins=5,multiple='stack', stat='count')
plt.xlabel('Capital Gain')
plt.ylabel('Frequency')
plt.title('Histogram of Capital Gain by Salary')
plt.show()

#Looks very largely skewed. Mostly reported zero capital gain
count_greater_than_zero = (df['capital-gain'] > 0).sum()
print(count_greater_than_zero)

#2538 out of 30K. 8%

#Count plot - race by salary
ax = sns.countplot(x='race', hue='salary', data=df)
plt.title('Count Plot of race by salary')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, int(height), ha="center")

plt.show()

#Predominantly white

#Count plot - race by salary
ax = sns.countplot(x='relationship', hue='salary', data=df)
plt.title('Count Plot of relationship by salary')

for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2., height + 0.1, int(height), ha="center")

plt.show()

#creating dataframe of dropped data
df_nan = raw.merge(df.drop_duplicates(),on='index',how='outer',indicator=True)

#find rows in raw df that dont match in df
df_nan = df_nan[df_nan['_merge']== 'left_only']

print(df_nan)

df_nan.to_csv('nan_data.csv')
df.to_csv('salary.csv')

#encode data to numerical values
""" enc = LabelEncoder()
col = df.columns[1:]
for x in range(len(col)):
    enc.fit(df[col[x]])
    df[col[x]] = enc.transform(df[col[x]])
df.to_csv('coded_salary.csv')    

 """

df = pd.read_csv('coded_salary.csv')
print(df)

# standardize and look at test distribution

#splitting y and x matrix and removing redundant & unnecessary columns
X = df.drop(columns=['salary', 'education','fnlwgt','index','Unnamed: 0'])
y = df['salary']

#split training and testing with an 80-20 split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20, random_state=1)

#Further split training set into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=1)

#training random forest model
rf = RandomForestClassifier(random_state=1)
rf_model = rf.fit(X_train, y_train)

#test baseline accuracy on dev set 
pred_rf = rf.predict(X_val)
print(str(accuracy_score(y_val, pred_rf)), ':RandForest')

#.85 accuracy score

#Cross-Validation
scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation Scores: ", scores)
print("Average CV Score: ", np.mean(scores))

#Average CV score: 0.848

#Variable importance 

#variable importance dataframe
importance = rf_model.feature_importances_
feature = rf_model.feature_names_in_
output = np.column_stack((importance, feature))
output = pd.DataFrame(output, columns=['importance','feature'])
output = output.sort_values(by=['importance'], ascending=False)
print(output)

#plot decrease in gini impurity
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=output, palette='viridis')
plt.title('Gini Importance (Mean Decrease Impurity)')
plt.show()

#calculate permutation importance (decrease in accuracy)
perm_importance = permutation_importance(rf_model, X_val, y_val, n_repeats=5, random_state=1)
perm_output = np.column_stack((perm_importance['importances_mean'], feature))
perm_output = pd.DataFrame(perm_output, columns=['importance','feature']
)
perm_output = perm_output.sort_values(by=['importance'], ascending=False)
print(perm_output)

#plot decrease in accuracy
plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=perm_output, palette='viridis')
plt.title('Permutation importance (decrease in accuracy)')
plt.show()

#confusion matrix 
cm = confusion_matrix(y_val, pred_rf)
print("Confusion Matrix:\n", cm)

tn, fp, fn, tp = confusion_matrix(y_val, pred_rf).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")

#Precision: 0.7372675828617623
#Recall: 0.614969656102495

#visualize using a heatmap 
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


#AUC
#gets probabilities for positive class (1)
y_prob = rf_model.predict_proba(X_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val,y_prob)
auc_score = roc_auc_score(y_val,y_prob)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='AUC = %0.2f' % auc_score)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Dashed diagonal line for random guessing
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='lower right')
plt.show()

print("AUC Score: ", auc_score)

#0.9 AUC score 

#Tuning hyperparameters/regularization 

#Tuning hyperparameters on validation set using GridSearchCV
param_grid = {
        'n_estimators': [100,200,300,500],
        'max_depth': [None, 10,20],
        'min_samples_split':[2,5,10],
        'max_features':['sqrt','log2'],
}

grid_search = GridSearchCV(estimator = rf, param_grid=param_grid, cv=3, n_jobs=-1)

#Fit model to the data
rf_best_fit = grid_search.fit(X_train,y_train)

#Get the best parameters and estimator
best_params = grid_search.best_params_
best_rf = grid_search.best_estimator_

print(best_rf)

#max depth = 20, min_samples_split = 10, n_estimators=300

#Make predictions based on best parameters and evaluate
y_pred = best_rf.predict(X_val)
print(str(accuracy_score(y_val, y_pred)), ':RandForest')

#0.867 accuracy score

#XGBoost - Ensemble method
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test)

#Parameter dictionary
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error'
}

#Define parameter grid for tuning
param_grid = {
    'max_depth': [3,5,7,10,20],
    'eta': [0.1,0.09,0.08],
    'n_estimators': [100,200,300,500],
    'subsample': [0.8, 0.7, 0.6, 0.5],
    'colsample_bytree': [0.8,0.7,0.6,0.5],
    'lambda': [1e-5,1e-2,0.1,1,100],
    'gamma': [0,0.1,0.2,0.3]
}

# Initialize the model
xgb_model = xgb.XGBClassifier(**params)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_grid,
    n_iter=50,  # Number of parameter settings sampled
    scoring='accuracy', 
    cv=3,  
    verbose=1,
    random_state=1,
    n_jobs=-1
)

# Fit to training data
random_search.fit(X_train, y_train)

# Get best parameters and best model
best_params = random_search.best_params_
best_model = random_search.best_estimator_
print("Best parameters found: ", best_params)

#Fitting 3 folds for each of 50 candidates, totalling 150 fits
#Best parameters found:  {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 7, 'lambda': 1, 'gamma': 0.1, 'eta': 0.08, 'colsample_bytree': 0.5}

y_pred = best_model.predict(X_val)
print(str(accuracy_score(y_val, y_pred)), ':XGBoost')

#0.867

#Re-Train best model with early stopping
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dval, 'validation')],
    early_stopping_rounds=10
)

#Predict on validation set & calculate accuracy
y_pred = final_model.predict(dval)
y_pred = [1 if pred > 0.5 else 0 for pred in y_pred]
print(str(accuracy_score(y_val, y_pred)), ':XGBoost')

#0.867 :XGBoost

#Plot feature importance
plt.figure(figsize=(20, 8))
xgb.plot_importance(final_model, importance_type='weight')  # Use 'weight', 'gain', or 'cover'
plt.title("Feature Importance")
plt.show()

#Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:\n", cm)

tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Precision: {precision}")
print(f"Recall: {recall}")

#Precision: 0.8003502626970228
#Recall: 0.6163182737693864

#visualize using a heatmap 
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Positive", "Negative"], yticklabels=["Positive", "Negative"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

#Make predictions on the Test Set
xgb_test_pred = final_model.predict(X_test)
rf_test_pred = best_rf.predict(X_test)

#Accuracy
xgb_accuracy = accuracy_score(y_test, xgb_test_pred)
rf_accuracy = accuracy_score(y_test, rf_test_pred)

print(str(xgb_accuracy),':XGB Accuracy')
print(str(rf_accuracy),':RF Accuracy')

#0.872037129123156 :XGB Accuracy
#0.8619260732637162 :RF Accuracy

#F1 Score
xgb_f1 = f1_score(y_test, xgb_test_pred, average='weighted')
rf_f1 = f1_score(y_test, rf_test_pred, average='weighted')

print(str(xgb_f1),':XGB F1 Score')
print(str(rf_f1),':RF F1 Score')

#0.8662585298596313 :XGB F1 Score
#0.8557120753133424 :RF F1 Score

#AUC Score
xgb_roc_auc = roc_auc_score(y_test, xgb_test_pred) 
rf_roc_auc = roc_auc_score(y_test, rf_test_pred) 

print(str(xgb_roc_auc),':XGB AUC Score')
print(str(rf_roc_auc),':RF AUC Score')

#0.7952878616709721 :XGB AUC Score
#0.7821529361818483 :RF AUC Score

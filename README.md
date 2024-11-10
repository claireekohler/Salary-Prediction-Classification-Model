# Salary-Prediction-Classification-Model

# Problem Statement
The overarching goal of my project was to utilize data extracted from the 1994 Census database to analyze how demographic indicators might correlate with salary earnings, specifically, whether an individual earns over or under $50,000 per annum. I wanted to answer the following questions:
-  What specific demographic indicators are most influential when understanding how much money someone makes?
-  What demographic information can lead to accurate predictions of salary?
-  What are the least important indicators of salary?

## Proposed Methodology
My data in its raw form had a few inconsistencies that had to be dealt with before it was usable as training and testing data. The raw data was made up of 32561 rows and 15 columns. After a quick glance at the data I found that there were many rows that had null values. My dataset was very large so I decided to go with a drop-N/A cleaning approach. This essentially means that I removed all rows that had an inconsistency instead of trying to fix them to be used. After removing all affected rows, the cleaned dataset was 30,162 rows. The removed rows were collected into a dataframe where I was able to confirm that there were no overt patterns to the removed data. I also decided to remove the final weight column which is a calculated value done by the census bureau. I did not want a calculated value to add unknown effects into my model so I decided to remove this column. I also removed the education column which was redundant due to the education num column also provided. After the cleaning was completed, I encoded the factored variables into numeric values that could be used to train models.

I used the 60-20-20 split rule of thumb – 60% training, 20% validation, and 20% testing data. 

I decided to implement a Random Forest analysis and was able to see which features are the most, and least, impactful on the outcome. I then decided to implement the powerful XGBoost Algorithm to see its performance on the dataset. After evaluating both model’s performance on the validation set, I tested them on the test set. The highest performing model was then deployed on a completely new census bureau set.

## Analysis and Results

Before fitting a model, I wanted to understand the spread of my data more, so I performed some exploratory data analysis. I discovered some of the following trends:

I found that more capital gain meant a higher salary, which isn’t quite a profound discovery and is what one probably expects. The distribution is also greatly skewed toward individuals with no capital gain, so that raises concern of inaccurate model predictions. Individuals not predicting capital gain is a common occurrence in census reports due to its complexity and people’s uncertainty.

<img width="501" alt="Screenshot 2024-11-09 at 17 09 26" src="https://github.com/user-attachments/assets/a50d1c0b-f6c0-4451-bd13-24d11c9274d8">

The kernel density plot of years of education revealed that the people earning more than 50K primarily have more than 10 years of education. It is unclear if this means post-high school, but for the sake of the problem, we will assume so. Age seems fairly normally distributed. 

<img width="443" alt="Screenshot 2024-11-09 at 17 10 17" src="https://github.com/user-attachments/assets/f56425cf-0451-49bc-860f-99020f57461f">

The histogram below reveals the dataset’s skewed distribution toward white men.

<img width="373" alt="Screenshot 2024-11-09 at 17 10 55" src="https://github.com/user-attachments/assets/a6f8c917-3484-4837-a3bc-bb89fa944118">

I decided to choose the Random Forest model because of its robustness and ability to determine which predictors are important in its predictions.

I analyzed my important features in two ways to get a better understanding of what features were impacting the predictions the most: Gini importance (mean decrease in impurity) and Permutation Importance (decrease in accuracy). 

<img width="461" alt="Screenshot 2024-11-09 at 17 11 41" src="https://github.com/user-attachments/assets/125ad673-e9f7-4006-b549-4ad22f12499d">

The gini importance tells us how much each feature is used in splits while the permutation importance tells us what features are critical in making accurate predictions. It’s useful to know that age is used a lot in splits, but my intuition is thinking because it has many distinct values. 

A few things caught my eye when looking at the permutation importance graph. Sex and native-country have a negative feature importance, which happens when the predictions on the shuffled data are more accurate than the real data. This occurs because the feature does not have high predictive power, but random chance causes the predictions on the shuffled data to be more accurate. I also thought it was interesting that race and sex did not contribute strongly to distinguishing patterns in my model. Since capital gain has extreme outliers, I believe the noise might be wrongly influencing the feature’s importance by creating “big splits” in some trees.

Performing a baseline cross-validation on the training set against the validation set to see how well the model generalizes across data, I received an 84.8% mean CV score. To measure the accuracy of my baseline model on this skewed dataset, I created a confusion matrix to gain insights into class-specific performance, in addition to using cross-validation to ensure consistent evaluation. The model has 61% recall and 73% precision. 73% accuracy suggests that the model is fairly good at minimizing false positives and is likely more conservative when predicting positives. 61% recall suggests the model has a tendency to miss some positive cases.

<img width="387" alt="Screenshot 2024-11-09 at 17 13 18" src="https://github.com/user-attachments/assets/b2ad92bc-ee9c-4077-9b48-e88107fd1cfe">

The random forest received a 0.9 AUC score, which portrays a fairly good classifier.

<img width="347" alt="Screenshot 2024-11-09 at 17 13 24" src="https://github.com/user-attachments/assets/3073b537-2eaf-42e7-a765-a031c932873c">

To check that my model was generalizing well across different parts of my data and reduce overfitting as much as possible, I tuned the hyperparameters using GridSearchCV of the random forest model. Grid search’s best parameters went as follows: max depth = 20, min_samples_split = 10, n_estimators=300. The accuracy score came out slightly better with a value of 0.867. This model is quite complex and concerned with overfitting. Although the model generalizes fairly well across the dataset, I’m curious if another model, like a boosting algorithm, will perform better. 

I then fit the data to the XGBoost algorithm, a boosting ensemble method. This algorithm also minimizes overfitting, but also increases speed, reduces variance, and leverages a learning rate.

After putting my data into the DMatrices, I performed RandomSearchCv for hyperparameter tuning because it's nonexhaustive and better to use with lengthy hyperparameter sets. The best parameters were as follows: Fitting 3 folds for each of 50 candidates, totalling 150 fits. Best parameters found:  {'subsample': 0.8, 'n_estimators': 100, 'max_depth': 7, 'lambda': 1, 'gamma': 0.1, 'eta': 0.08, 'colsample_bytree': 0.5}. 
After re-training the best model with early stopping, the final model received an accuracy score of 86.7% on the validation set. 

Using XGBoost’s built-in feature importance, I plotted the gain, which measures the improvement in accuracy (reduction in impurity) from splits, for each feature. Surprisingly marital status was near the top, which is different to my results from my Random Forest model.

<img width="1679" alt="Screenshot 2024-11-08 at 14 59 40" src="https://github.com/user-attachments/assets/011cd995-eef5-4db5-9270-347c3872ae3c">

<img width="381" alt="Screenshot 2024-11-09 at 17 16 09" src="https://github.com/user-attachments/assets/5a7d5c31-48a4-4263-b3ad-fbd0b9bc9682">

The XGBoost model had 80% precision and 62% recall. It had a 10% higher chance at minimizing false positives than the Random Forest model.

I made predictions for both the Random Forest and XGBoost model on the test set and evaluated them using accuracy, F1, and AUC scores. I weighed more of my interpretation on F1 because it serves as a better metric when dealing with imbalanced classes such as this case. 
XGBoost received a higher score on all three metrics by a consistent 0.1. XGBoost had an 0.87 F1 score and Random Forest had an 0.86 F1 score. 

I found that while both Random Forest and XGBoost were average classifiers, XGBoost was slightly more accurate in its predictions. Capital gain, education, and relationship were the three features, in both Random Forest and XGBoost, that had the highest importance in the model’s predictive power. 

## Conclusions and Further Considerations
In aiming to answer what demographic indicators are most important, I found that the important features were capital gain, education, and relationship, and age in the Random Forest model and relationship, capital gain, education, and marital status were the most important features in the XGBoost model. The least important factors were consistently native country, race, and sex. XGBoost ended up being the best analysis method for my dataset, most likely because of its ability to handle complexity and imbalance data more effectively. Since this census had mostly white and mostly male respondents, race and gender end up being the least important indicators, since there’s such a large skew and relatively few data points to work with outside of the majority demographic. Thus, I found that algorithmic prediction weighted race and gender less significantly and resulted in a more accurate prediction model.

The demographic tilt of the dataset was certainly something I had to account for in the design and execution of my statistical analysis, and thus if I are to extrapolate these methods for a more modern census dataset, I might find I’d have to alter the analysis approach and account for more diverse data. Going forward, I might also want to look at the lesser represented demographics and see how accurate my predictions are. If I are to take out all the white males in the dataset, how much would the prediction accuracy change across the analysis methods? Could I retrain the algorithms to predict on this smaller dataset? Furthermore, if I were to compare the results of this census data and the most recent 2020 one, what might I find to be the difference in the significance of factors like sex and race? Finally, what demographic indicators would I want to add to these datasets to have a better understanding of demographic indicators of income?

Additional out of scope items for later consideration: 
- How the generalization error decreases with the number of trees run. 
- The use of SHAP values to interpret feature importance
- Different data cleaning approach, specifically data imputation.

There’s a lot that can be learned from an exploration of demographic variables and predicting income outcomes, not just in my analysis but in the broader context of economics, sociology, and politics as a whole. I learned that demographic information can yield a powerful prediction model of someone’s income level at a binary level (in this case, more or less than $50,000), as well as how to apply learning models to achieve this outcome. I also learned how to pull insights from a large dataset and dive deeper into the dataset and its factors.















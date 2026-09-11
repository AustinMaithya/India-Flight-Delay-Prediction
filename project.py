
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import shutil
import warnings 
warnings.filterwarnings('ignore')

#Unzipping the file into a csv file

shutil.unpack_archive('C:\\Users\\HP\\Desktop\\Machine Learning Projects\\India Flight Delay Prediction\\indian-flight-delay-datasets.zip', 'flight_delay.csv')

#Loading the dataset
df = pd.read_csv("C:\\Users\\HP\\Desktop\\Machine Learning Projects\\India Flight Delay Prediction\\flight_delay.csv\\india_flight_delay_model_input.csv")
df.head()


#Checking the data information
df.info()


#converting the columns to lower case
df.columns = df.columns.str.lower()
df.columns


#converting the date columns to datetimeformat
df['flight_date'] = pd.to_datetime(df['flight_date'])





df.head()


#Checking if the conversions are okay
df.info()


df['previous_flight_delay_minutes'].value_counts()


#checking data descriptions
df.describe()


#checking data shape
df.shape


#Checking the correlation of the numerical columns 

cols =[
       'scheduled_departure_hour',
       'scheduled_departure_minute', 'day_of_week', 'month', 'is_weekend',
       'peak_hour',  'temperature_c', 'humidity_pct',
       'wind_speed_kmh', 'visibility_km', 'rainfall_mm', 'cloud_cover_pct',
       'origin_congestion_index', 'previous_flight_delay_minutes',
       'turnaround_risk_index']

corr = df[cols].corr()
corr


#heatmap of the corellation matrix
fig,ax = plt.subplots(figsize=(16,8))
sns.heatmap(corr, annot=True, cmap='coolwarm')

ax.set_title('Correlation Matrix', fontsize=20)
plt.show()


df.head()


df.columns


df.groupby('airline')['previous_flight_delay_minutes'].aggregate(['mean','median'])


fig,ax = plt.subplots(figsize=(16,8))
sns.scatterplot(data=df, 
             x='flight_date', 
             y='previous_flight_delay_minutes', 
             hue='delay_target')

ax.set_xlabel('Date')
ax.set_ylabel('Previous Flight Delay (minutes)')
ax.title.set_text('Previous Flight Delay Over Time by Airline')


plt.tight_layout()
plt.show()


#Imporing machine learning libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA




df.info()


df.head()


#Splitting into dependent and Independent variable
X = df.drop(columns=['delay_target', 'flight_date'])
y = df['delay_target']



#splitting into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Splitting into categorical and numerical columns
categorical_cols = ['airline',
                    'weather',
                    'origin_airport',
                    'destination_airport' ]

numerical_cols = df.drop(columns=categorical_cols + ['delay_target', 'flight_date','flight_number','departure_delay']).columns.tolist()




preprocessing = ColumnTransformer(
    transformers=[("num", StandardScaler(), numerical_cols),
                   ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                   ])



models = {
    "LogisticRegression": LogisticRegression(), 
    # "SVM": SVC(),   
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "XGB": xgb.XGBClassifier()
    
}


def train_test_evaluate_models(models, preprocessing, X_train, y_train, X_test, y_test):
    results = {}    

    for name,model in models.items():


        pipeline = Pipeline([("preprocessing", preprocessing),
                                    ("model", model)])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)



        results[name]= {
                "classification_report" : classification_report(y_test,y_pred),
                "accuracy_score" :accuracy_score(y_test,y_pred),
                "confusion_matrix" :confusion_matrix(y_test,y_pred),
                "precision_score" :precision_score(y_test,y_pred),
                "recall_score":recall_score(y_test,y_pred),
                "f1_score":f1_score(y_test,y_pred)
            }
        print(f"{name}")
        print(results[name])

    return results





results = train_test_evaluate_models(models, 
                                     preprocessing, 
                                     X_train, 
                                     y_train, 
                                     X_test,
                                     y_test)
output = pd.DataFrame(results)


#Performing cross validation to check and confirm the evaluations of the models
def evaluate_model_with_cross_validation(model, preprocessing, X, y):
    pipeline = Pipeline([("preprocessing", preprocessing),
                         ("model", model)])
    cross_validation = cross_val_score(pipeline, X, y, cv=5, scoring='f1')
    return cross_validation

print("Cross Validation Results:")
for name, model in models.items():
    cv_results = evaluate_model_with_cross_validation(model, preprocessing, X, y)
    print(f"{name}: {cv_results.mean():.3f} (+/- {cv_results.std() * 2:.3f})")


#Since XGBClassifier performed the best, we will perform hyperparameter tuning on it to improve its performance
# xgb.XGBClassifier().get_params()

param_grid = {
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__n_estimators': [100, 200, 300],
    'model__subsample': [0.8, 1.0],
    'model__colsample_bytree': [0.8, 1.0]
}

# Hyper parameter tuning for XGBoost Classifier

XGB_GridSearch = GridSearchCV(
    estimator=Pipeline([("preprocessing", preprocessing),
                        ("model", xgb.XGBClassifier())]),
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=2
)


XGB_GridSearch.fit(X_train, y_train)



#Creating an output of the best parameters and best model from the hyperparameter tuning
print("Best Parameters:", XGB_GridSearch.best_params_)
best_model = XGB_GridSearch.best_estimator_


#Predicting the test data using the best model
y_pred_xgb = best_model.predict(X_test)


#Evaluating the best model using classification report
print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))


#Further evaluation of the best model using accuracy, precision, recall and f1 score
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("Precision:", precision_score(y_test, y_pred_xgb))
print("Recall:", recall_score(y_test, y_pred_xgb))
print("F1 Score:", f1_score(y_test, y_pred_xgb))


#Visualizing the confusion matrix
cm = confusion_matrix(y_test, y_pred_xgb)

sns.heatmap(
    cm,
    annot=True,
    fmt='d'
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")
plt.show()


#Further evaluation of the best model using ROC-AUC score
from sklearn.metrics import roc_curve, roc_auc_score
y_prob_xgb = best_model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(y_test,y_prob_xgb)

print("\nROC-AUC:", roc_auc)


#Plotting the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob_xgb)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f'XGBoost (AUC = {roc_auc:.3f})'
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle='--',
    label='XGBoost Classifier'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Tuned XGBoost')
plt.legend()
plt.show()


#Exporting the best model to a pickle file for future use
import joblib
joblib.dump(best_model, 'Indian_Flight_Delay_Model.pkl')

# %%
#getting the probability of every flight
y_pred_xgb =  best_model.predict(X_test)
#getting the probability 
y_prob_xgb = best_model.predict_proba(X_test)[:, 1]
#place the predictions in a dataframe
prediction_df = X_test.copy()
prediction_df["actual"] = y_test
prediction_df["prediction"] = y_pred_xgb
prediction_df["delay_probability"] = y_prob_xgb
prediction_df.head()

#create an identifier 
flight_ids = df.loc[X_test.index, "flight_number"]
prediction_df["flight_number"] = flight_ids


#Converting probability to risk
def classify_risk(probability):

    if probability < 0.30:
        return "LOW"

    elif probability < 0.60:
        return "MEDIUM"

    elif probability < 0.80:
        return "HIGH"

    else:
        return "VERY HIGH"


prediction_df["risk"] = prediction_df["delay_probability"].apply(
    classify_risk
)

prediction_df["probability_pct"] = (
    prediction_df["delay_probability"] * 100
).round(1)



flight_monitor = prediction_df[
    [
        "flight_number",
        "probability_pct",
        "risk"
    ]
].copy()

flight_monitor = flight_monitor.sort_values(
    "probability_pct",
    ascending=False
)
flight_monitor.head(10)


#Getting the main risk drivers 
xgb_model = best_model.named_steps["model"]
preprocessor = best_model.named_steps["preprocessing"]
feature_names = preprocessor.get_feature_names_out()
importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})
importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)
importance_df.head(10)




#Barchart on the above 
top_features = importance_df.head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top Factors Used by XGBoost")

plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()


#recommendation 
def recommend_action(risk):

    if risk == "LOW":
        return "Normal operations"

    elif risk == "MEDIUM":
        return "Monitor flight"

    elif risk == "HIGH":
        return "Conduct operational review"

    elif risk == "VERY HIGH":
        return "Immediate operational review"


prediction_df["recommendation"] = (
    prediction_df["risk"].apply(recommend_action)
)


#recommended task
def recommend_action(row):

    if row["delay_probability"] >= 0.80:

        if row["turnaround_risk_index"] > 0.70:
            return "Prioritize turnaround review"

        elif row["origin_congestion_index"] > 0.70:
            return "Review congestion and gate availability"

        elif row["previous_flight_delay_minutes"] > 30:
            return "Monitor inbound aircraft"

        else:
            return "Immediate operational review"

    elif row["delay_probability"] >= 0.60:
        return "Monitor flight closely"

    elif row["delay_probability"] >= 0.30:
        return "Routine monitoring"

    else:
        return "Normal operations"


prediction_df["recommendation"] = prediction_df.apply(
    recommend_action,
    axis=1
)


total_flights = len(prediction_df)

high_risk_flights = prediction_df[
    prediction_df["risk"].isin(["HIGH", "VERY HIGH"])
].shape[0]

average_probability = (
    prediction_df["delay_probability"].mean() * 100
)

print("Total Flights:", total_flights)

print("High Risk Flights:", high_risk_flights)

print(
    "Average Delay Probability:",
    round(average_probability, 2),
    "%"
)


dashboard_table = prediction_df[
    [
        "flight_number",
        "probability_pct",
        "risk",
        "recommendation"
    ]
].sort_values(
    "probability_pct",
    ascending=False
)

dashboard_table.head(20)



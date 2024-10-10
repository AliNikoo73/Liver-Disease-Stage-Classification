from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shap
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    KFold,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
import seaborn as sns
import os

# Restrict plot formats to 'jpg' and 'png'
formats = ["jpg", "png"]
base_dir = "Projects/liver disease detection/results/saved_plots"

# Create directories for the selected formats
for fmt in formats:
    os.makedirs(os.path.join(base_dir, fmt), exist_ok=True)

# Function to save plots in 'jpg' and 'png' formats only
def save_all_formats(fig, plot_name):
    for fmt in formats:
        fig.savefig(os.path.join(base_dir, fmt, f"{plot_name}.{fmt}"), format=fmt)

# Function to remove outliers based on IQR
def remove_outliers_iqr(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

import matplotlib.pyplot as plt
import seaborn as sns

# Function to plot box plots for each feature in X
def plot_box_plots(df):
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x=column)
        plt.title(f'Box Plot of {column}')
        plt.tight_layout()
        # plt.show()
        for fmt in ["jpg"]:
            plt.savefig(f'Projects/liver disease detection/results/saved_plots/box_plots{column}.{fmt}', format=fmt)
    plt.close()


# Load dataset & preprocess
df = pd.read_csv("Projects/liver disease detection/data/archive/liver_cirrhosis.csv")
# print(df.isna().sum()) # Check for missing values
X = df.drop(columns=["Stage", "N_Days", "Status", "Drug", "Edema", "Sex"])
y = df["Stage"]
num_cols = X.select_dtypes(include=[np.number]).columns # Select numeric columns
bool_cols = X.select_dtypes(include=[np.object_]).columns # Select boolean columns
X[bool_cols] = X[bool_cols].astype("bool") # Convert string value columns to boolean objects

# print(X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Split data into training and testing sets

# Call the function to plot box plots for all features in X
# plot_box_plots(X_train)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[num_cols]), columns=num_cols)
X_test_scaled = pd.DataFrame(scaler.transform(X_test[num_cols]), columns=num_cols)

# Merge scaled numeric & boolean columns, impute NaNs with mode
X_train_final = pd.concat([X_train_scaled, X_train[bool_cols].reset_index(drop=True)], axis=1)
X_train_final.fillna(X_train_final.mode().iloc[0], inplace=True)
X_test_final = pd.concat([X_test_scaled, X_test[bool_cols].reset_index(drop=True)], axis=1)
X_test_final.fillna(X_train_final.mode().iloc[0], inplace=True)

# Feature importance using RandomForest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_final, y_train)
imp_features = pd.Series(rf.feature_importances_, index=X_train_final.columns).sort_values(ascending=False)
important_features = imp_features[imp_features > 0.05].index
excluded_features = X_train_final.columns.difference(important_features).tolist()

# Grid search with cross-validation
param_grid = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1,
)
grid_search.fit(X_train_final[important_features], y_train)

# Best model & performance evaluation
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_final[important_features])

# Metrics calculation
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred, average="weighted"),
    "Recall": recall_score(y_test, y_pred, average="weighted"),
    "F1 Score": f1_score(y_test, y_pred, average="weighted"),
}

# Output feature importances and performance
print(f"Feature importances:\n{imp_features}")
print(f"Selected important features: {important_features.tolist()}")
print(f"Best parameters: {grid_search.best_params_}")
print("\nPerformance on Test Set:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Assuming 'important_features' is a Pandas Index and 'grid_search.best_params_' is a dictionary
important_features_list = important_features.tolist()
best_params_dict = grid_search.best_params_

# Creating a DataFrame to store important features
important_features_df = pd.DataFrame({
    'Selected Important Features': important_features_list
})

# Creating a DataFrame to store best parameters
best_params_df = pd.DataFrame({
    'Parameter scale before removing outliers': best_params_dict.keys(),
    'Best Value': best_params_dict.values()
})

# Saving both DataFrames to a single CSV
with open('Projects/liver disease detection/results/model_scale before outlier removal_info.csv', 'w') as f:
    # Write the important features first
    important_features_df.to_csv(f, index=False)
    
    # Add a blank row to separate the sections
    f.write("\n")

    # Write the best parameters next
    best_params_df.to_csv(f, index=False)

print("Model info saved to 'model_scale before outlier removal_info.csv'")

# 1. Correlation Heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(X_train_final.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap Before removing outliers")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
save_all_formats(fig, "correlation_heatmap_Before_removing_outliers")
# plt.show()

# 2. Cumulative Feature Importance
fig, ax = plt.subplots(figsize=(8, 6))
imp_features_cum = imp_features[~imp_features.index.isin(excluded_features)].cumsum()
ax.plot(imp_features_cum, marker="o")
ax.set_title("Cumulative Feature Importance After scaling Before removing outliers")
ax.set_xlabel("Feature")
ax.set_ylabel("Cumulative Importance")
plt.xticks(rotation=90)
plt.tight_layout()
save_all_formats(fig, "Cumulative Feature Importance After scaling Before removing outliers")
# plt.show()

# 3. Performance Metrics Bar Plot
fig, ax = plt.subplots(figsize=(8, 6))
metric_names = list(metrics.keys())
metric_values = list(metrics.values())
ax.barh(metric_names, metric_values, color="skyblue")
ax.set_xlim(0, 1)
ax.set_xlabel("Score")
ax.set_title("Test Set Performance After Scaling Before removing outliers")
for index, value in enumerate(metric_values):
    ax.text(value, index, f"{value:.2f}")
plt.tight_layout()
save_all_formats(fig, "Test Set Performance After Scaling Before removing outliers")
# plt.show()

# 4. Pair Plot of Important Features
fig = sns.pairplot(
    pd.concat([X_train_final[important_features], y_train.reset_index(drop=True)], axis=1),
    hue="Stage",
)
fig.fig.suptitle("Pairplot of Important Features with Target After scaling Before removing outliers", y=1.02)
save_all_formats(fig, "Pairplot of Important Features with Target After scaling Before removing outliers")
# plt.show()

print("\n" + "==" * 50)
print("Results saved for first stage...")
print("now move on to next stage where we remove outliers and then scale and then train")
print("\n" + "==" * 50)

# Outlier Removal and Retraining on initial dataset
outlier_columns = X.drop(columns = ["Age", "Ascites", "Hepatomegaly", "Spiders"]).columns
X_Second = X[outlier_columns]

# Train-test split
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_Second, y, test_size=0.2, random_state=42) # Split data into training and testing sets

X_train_final_no_outliers = remove_outliers_iqr(X_train2, outlier_columns)
X_train_final_no_outliers.reset_index(drop=True, inplace=True)
y_train_no_outliers = y_train2.iloc[X_train_final_no_outliers.index]

# Split the cleaned training data into train and validation sets
X_train_clean, X_val_clean, y_train_clean, y_val_clean = train_test_split(
    X_train_final_no_outliers, y_train_no_outliers, test_size=0.2, random_state=42
)

# Grid search with cross-validation
param_grid2 = {
    "n_estimators": [50, 100, 150],
    "max_depth": [5, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search2 = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid2,
    cv=kfold,
    scoring="accuracy",
    n_jobs=-1,
)


# Train the model using GridSearchCV on the cleaned data
grid_search2.fit(X_train_clean, y_train_clean)

# Best model and evaluation
best_rf_clean = grid_search2.best_estimator_
y_val_pred = best_rf_clean.predict(X_val_clean)

# Metrics calculation for cleaned data
metrics_cleaned = {
    "Accuracy": accuracy_score(y_val_clean, y_val_pred),
    "Precision": precision_score(y_val_clean, y_val_pred, average="weighted"),
    "Recall": recall_score(y_val_clean, y_val_pred, average="weighted"),
    "F1 Score": f1_score(y_val_clean, y_val_pred, average="weighted"),
}

print("Performance on Validation Set (Cleaned Data):")
for metric, value in metrics_cleaned.items():
    print(f"{metric}: {value:.4f}")

# Assuming 'grid_search.best_params_' is a dictionary
best_params_dict2 = grid_search2.best_params_

# Creating a DataFrame to store best parameters
best_params_df = pd.DataFrame({
    'Parameter scale after removing outliers': best_params_dict2.keys(),
    'Best Value': best_params_dict2.values()
})

# Saving both DataFrames to a single CSV
with open('Projects/liver disease detection/results/model_scale after outlier removal_info.csv', 'w') as f:
    # Write the important features first
    important_features_df.to_csv(f, index=False)
    
    # Add a blank row to separate the sections
    f.write("\n")

    # Write the best parameters next
    best_params_df.to_csv(f, index=False)

print("Model info saved to 'model_scale after outlier removal_info.csv'")

# Feature Importance and Performance Metrics After Outlier Removal
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].barh(imp_features.index, imp_features, color="skyblue")
ax[0].invert_yaxis()
ax[0].set_xlabel("Importance")
ax[0].set_title("Feature Importance After removing outliers then scaling")

ax[1].barh(list(metrics_cleaned.keys()), list(metrics_cleaned.values()), color="skyblue")
ax[1].set_xlim(0, 1)
ax[1].set_xlabel("Score")
ax[1].set_title("Validation Set Performance After removing outliers then scaling")
plt.tight_layout()
plt.show()

# 1. Correlation Heatmap After Outlier Removal
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(X_train_clean.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Feature Correlation Heatmap After removing outliers then scaling")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
save_all_formats(fig, "correlation_heatmap_After_removing_outliers then scaling")
# plt.show()

# 2. Cumulative Feature Importance After Outlier Removal
fig, ax = plt.subplots(figsize=(8, 6))
imp_features_cum = imp_features[~imp_features.index.isin(excluded_features)].cumsum()
ax.plot(imp_features_cum, marker="o")
ax.set_title("Cumulative Feature Importance After removing outliers then scaling")
ax.set_xlabel("Feature")
ax.set_ylabel("Cumulative Importance")
plt.xticks(rotation=90)
plt.tight_layout()
save_all_formats(fig, "cumulative_feature_importance_After_removing_outliers then scaling")
# plt.show()

# 3. Performance Metrics Bar Plot After Outlier Removal
fig, ax = plt.subplots(figsize=(8, 6))
metric_names = list(metrics_cleaned.keys())
metric_values = list(metrics_cleaned.values())
ax.barh(metric_names, metric_values, color="skyblue")
ax.set_xlim(0, 1)
ax.set_xlabel("Score")
ax.set_title("Validation Set Performance After removing outliers then scaling")
for index, value in enumerate(metric_values):
    ax.text(value, index, f"{value:.2f}")
plt.tight_layout()
save_all_formats(fig, "performance_metrics_After_removing_outliers then scaling")
# plt.show()

# Pair Plot After Outlier Removal
fig = sns.pairplot(
    pd.concat([X_train_clean[important_features], y_train_clean.reset_index(drop=True)], axis=1),
    hue="Stage",
)
fig.fig.suptitle("Pairplot of Important Features with Target After removing outliers then scaling", y=1.02)
save_all_formats(fig, "pair_plot_After_removing_outliers then scaling")
# plt.show()
plt.close()
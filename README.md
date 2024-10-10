# 🧬 **Liver Disease Detection Using RandomForest and Feature Importance Analysis**

---

## 📜 **Summary**

This project focuses on building a machine learning model to predict **liver disease stages** using clinical data. The workflow involves:

- **Preprocessing the dataset**: Handling missing values, scaling numerical features, and removing outliers to improve model performance.
- **Feature selection**: Employing the **RandomForest** algorithm for feature selection and model training.
- **Hyperparameter tuning**: Utilizing **GridSearchCV** for optimization.
- **Model evaluation**: Using metrics like **accuracy, precision, recall, and F1 score** to assess the model's performance.

Various **visualizations** such as **feature importance plots**, **correlation heatmaps**, and **box plots** are generated to offer insights into the data and model behavior.

---

## 🎯 **Objective**
> To develop a machine learning model that accurately predicts liver disease stages by leveraging RandomForest and feature importance analysis, improving clinical decision-making.

---

## 🛠 **Skills Required**

### **Technical Skills**

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/-Seaborn-3776AB?style=for-the-badge&logoColor=white)

- **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
- **Data Preprocessing and Cleaning** (Handling missing values, outliers)
- **Machine Learning** (RandomForest, GridSearchCV, KFold Cross-Validation)
- **Model Performance Evaluation** (Accuracy, Precision, Recall, F1 Score)
- **Data Visualization** (Box plots, Heatmaps, Feature Importance plots)

### **Soft Skills**

- 🔍 **Problem-Solving & Critical Thinking**
- 🎯 **Attention to Detail**
- ⏱️ **Time Management**

---

## 📊 **Deliverables**

### **Key Outputs**

- 🧪 **Preprocessed Liver Disease Dataset**: Missing values imputed, outliers removed.
- 🌲 **Trained RandomForest Model**: Optimized hyperparameters through GridSearchCV.
- 📊 **Feature Importance Analysis**: Important features selected for prediction.
- 📈 **Performance Metrics Report**: Including accuracy, precision, recall, and F1 score.

### **Visualizations**
- 🔥 **Correlation Heatmap**
- 📦 **Box Plots** of Features
- 🌟 **Feature Importance Plots**
- 📉 **Performance Metrics Bar Plots**
- 🎨 **Pair Plots** of Important Features with the Target Variable

---

## 🔍 **Additional Information**

- **Dataset Source**: [Liver Cirrhosis Stage Classification](https://www.kaggle.com/datasets/aadarshvelu/liver-cirrhosis-stage-classification)
- **Preprocessing**: Applied **data scaling**, **outlier removal** using **Interquartile Range (IQR)**, and **missing value imputation** using the mode.
- **Model Selection**: RandomForest was chosen for its ability to rank feature importance.
- **Hyperparameter Tuning**: **GridSearchCV** used for optimal model configuration.
- **Interpretability**: Emphasis on model interpretability using **cumulative feature importance** to provide insights into the most influential features for predicting liver disease stages.

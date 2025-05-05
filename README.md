# 🧬 **Liver Disease Detection Using RandomForest and Feature Importance Analysis**

[![CI](https://github.com/AliNikoo73/Liver-Disease-Stage-Classification/actions/workflows/ci.yml/badge.svg)](https://github.com/AliNikoo73/Liver-Disease-Stage-Classification/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/AliNikoo73/Liver-Disease-Stage-Classification/branch/main/graph/badge.svg)](https://codecov.io/gh/AliNikoo73/Liver-Disease-Stage-Classification)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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

## Features

- Data preprocessing and cleaning
- Feature selection using RandomForest importance
- Model training with hyperparameter tuning
- Comprehensive evaluation metrics
- Visualization tools for model insights
- Unit tests with pytest
- CI/CD pipeline with GitHub Actions

## Installation

1. Clone the repository:

```bash
git clone https://github.com/AliNikoo73/Liver-Disease-Stage-Classification.git
cd Liver-Disease-Stage-Classification
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset as described in `data/README.md`

2. Run the example notebook:

```bash
jupyter notebook notebooks/liver_disease_classification.ipynb
```

3. Import and use the package in your own code:

```python
from src import data, features, model, evaluate

# Load and preprocess data
df = data.load_data('data/liver_cirrhosis.csv')
X, y = data.preprocess_data(df)

# Train model
best_model, best_params = model.train_model(X, y)

# Make predictions
predictions = model.predict(best_model, X)
```

## Project Structure

```
├── src/                    # Source code
│   ├── data.py            # Data loading and preprocessing
│   ├── features.py        # Feature selection and engineering
│   ├── model.py           # Model training and tuning
│   └── evaluate.py        # Evaluation and visualization
├── tests/                  # Unit tests
├── notebooks/             # Jupyter notebooks
├── data/                  # Dataset directory
├── results/               # Output directory for plots
├── .github/               # GitHub Actions workflows
├── requirements.txt       # Project dependencies
├── setup.py              # Package setup file
├── LICENSE               # MIT License
└── README.md             # This file
```

## Development

1. Install development dependencies:

```bash
pip install -e ".[dev]"
```

2. Run tests:

```bash
pytest tests/
```

3. Run linting:

```bash
black .
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

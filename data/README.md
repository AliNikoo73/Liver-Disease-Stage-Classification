# Data Directory

This directory contains the dataset used for liver disease stage classification.

## Dataset

The dataset used in this project is the Liver Cirrhosis dataset, which contains various clinical features and patient information for predicting liver disease stages.

### How to Obtain the Dataset

1. Visit the Kaggle dataset page: [Liver Cirrhosis Dataset](https://www.kaggle.com/datasets/liver-cirrhosis)
2. Download the `liver_cirrhosis.csv` file
3. Place the downloaded file in this directory

### Data Structure

The dataset contains the following features:

- Stage (Target variable)
- N_Days
- Status
- Drug
- Age
- Sex
- Ascites
- Hepatomegaly
- Spiders
- Edema
- Bilirubin
- Cholesterol
- Albumin
- Copper
- Alk_Phos
- SGOT
- Tryglicerides
- Platelets
- Prothrombin
- Stage

### Data Privacy

Note: This dataset is for research purposes only. Please ensure you comply with all applicable data protection and privacy regulations when using this dataset.

## .gitignore Configuration

The `.gitignore` file is configured to exclude large data files. The following patterns are ignored:

- \*.csv
- \*.xlsx
- \*.xls
- \*.parquet
- \*.feather
- \*.pickle
- \*.pkl

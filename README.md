# Music and Mental Health Analysis

## Overview

This project investigates the relationship between music listening habits and mental well-being. It analyzes survey data to understand how different musical preferences and listening behaviors correlate with self-reported levels of anxiety, depression, insomnia, and OCD. The goal is to gain insights into the potential effects of music on mental health.

## Data Source

The data for this analysis comes from a survey dataset located in `/kaggle/input/mxmh-survey-results/mxmh_survey_results.csv`.

## Libraries Used

The following Python libraries were used in this project:

* **pandas:** For data manipulation and analysis.
* **numpy:** For numerical operations.
* **scikit-learn (sklearn):** For preprocessing, model selection, evaluation, and machine learning algorithms (Naive Bayes, Neural Networks).
* **matplotlib.pyplot:** For basic data visualization.
* **seaborn:** For enhanced statistical data visualization.

## Data Exploration and Preprocessing

The initial steps involved:

1.  **Loading the data:** Reading the CSV file into a pandas DataFrame.
2.  **Initial inspection:** Examining the first few rows (`df.head()`), data types (`df.info()`), descriptive statistics (`df.describe()`), and the shape of the data (`df.shape`).
3.  **Missing value analysis:** Identifying and counting missing values in each column (`df.isnull().sum()`).
4.  **Unique value identification:** Determining the number of unique values in each column (`df.nunique()`).
5.  **Target variable analysis:** Examining the distribution of the 'Music effects' target variable (`df['Music effects'].value_counts()`).
6.  **Data cleaning:**
    * Creating a copy of the DataFrame to avoid modifying the original (`data = df.copy()`).
    * Dropping rows with any missing values (`data.dropna(axis='index', how='any', inplace=True)`).
    * Converting the 'Music effects' categorical variable into numerical representations (`data['Music effects'].map({'Improve':0,'No effect':1,'Worsen':2})`).
7.  **Feature visualization:** Plotting histograms for numerical features ('Age', 'Hours per day', 'BPM', 'Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects') to understand their distributions.
8.  **Pairwise relationships:** Generating a pair plot using seaborn (`sns.pairplot()`) to visualize relationships between different features, colored by the 'Music effects' category.
9.  **Feature engineering:**
    * Dropping the 'Timestamp' column as it's likely not relevant for the analysis (`data.drop(columns=['Timestamp'], inplace=True)`).
    * Converting categorical features into numerical representations using one-hot encoding with `pd.get_dummies(data)`.
10. **Correlation analysis:** Calculating and visualizing the correlation matrix (`data.corr()`, `sns.heatmap()`) to understand the relationships between different variables, including the target variable.
11. **Further data transformation:**
    * Creating another copy of the DataFrame (`T_data = df.copy()`).
    * Dropping rows with missing values (`T_data.dropna(...)`).
    * Converting relevant float columns ('Age', 'Anxiety', 'Depression', 'Insomnia', 'OCD') to integer types.
    * Renaming columns for better readability (`T_data.rename(...)`).
12. **Target variable distribution visualization:** Creating a bar plot to visualize the counts of each category in the 'effects' column (`T_data.value_counts(T_data["effects"]).plot(kind="bar")`).
13. **Checking for unexpected values:** Iterating through columns to check for any '?' values (`for value in T_data.columns: print(value,":", sum(T_data[value] == '?'))`).

## Model Development and Evaluation

The project explored several machine learning approaches:

### 1. Complement Naive Bayes (CNB)

* **Data Splitting:** The data was split into training and testing sets (`train_test_split`) with a test size of 33% and a random state of 5 for reproducibility.
* **Model Training:** A Complement Naive Bayes classifier (`CNB()`) was initialized and trained on the training data (`classifier.fit(X_train, Y_train)`).
* **Prediction:** Predictions were made on both the training and testing sets (`classifier.predict()`).
* **Evaluation:** The model's performance was evaluated using:
    * Accuracy score (`ACC_SC`).
    * Confusion matrix (`confusion_matrix`).
    * F1-score (`f1_score`) with different averaging methods (None, weighted, micro, macro).
    * Precision score (`precision_score`) with different averaging methods.
    * Recall score (`recall_score`) with different averaging methods.
    * Visualization of confusion matrices for both training and testing data using heatmaps (`sns.heatmap()`).
    * K-fold cross-validation (`cross_val_score`) to assess the model's generalization ability.

### 2. Complement Naive Bayes with Feature Selection and Hyperparameter Tuning

* **Feature Selection:** Some columns ('Timestamp', 'Primary streaming service', 'Permissions') were dropped from a copied DataFrame (`data_improving`) to potentially improve model performance.
* **Data Splitting:** The modified data was again split into training and testing sets.
* **Model Training:** A new Complement Naive Bayes classifier (`clf = CNB()`) was trained.
* **Hyperparameter Tuning:** GridSearchCV (`GridSearchCV`) was used to find the optimal hyperparameters for the CNB model. The parameters tuned were `alpha`, `fit_prior`, `norm`, and `class_prior`.
* **Best Model Evaluation:** The best model found by GridSearchCV was used to make predictions on the test set, and its performance was evaluated using accuracy and a classification report (`CLA_RE`). Confusion matrices for the best model on both training and testing data were also visualized.

### 3. Multi-layer Perceptron (MLP) Classifier (Neural Network)

* **Model Initialization:** An MLPClassifier (`MLPClassifier`) was initialized with specific hidden layer sizes, maximum iterations, activation function ('relu'), solver ('adam'), and a random state for reproducibility.
* **Model Training:** The MLP classifier was trained on the training data (`mlp.fit(train1, train2)`).
* **Prediction:** Predictions were made on both the training and testing sets (`mlp.predict()`).
* **Evaluation:** Similar to the CNB model, the MLP's performance was evaluated using accuracy, F1-score, precision, recall, and confusion matrix visualizations.
* **Cross-validation:** K-fold cross-validation was also performed on the MLP model.

## Results

The results of the different models are presented in the code output, including accuracy scores, confusion matrices, and classification reports. Further analysis and interpretation of these results would be needed to draw meaningful conclusions about the relationship between music and mental health based on this data.

## Author

Anoud ALfaydi

## BBB Permeability Prediction Project

This project aims to develop a machine learning model to predict the permeability of compounds through the Blood-Brain Barrier (BBB). The model classifies a given molecule as either BBB+ (permeable) or BBB- (non-permeable) based on its molecular descriptors.

***

### 🎯 Project Goal

The primary objective is to build a robust and accurate classification model using a dataset of molecular compounds and their known BBB permeability status. The project follows a standard machine learning workflow, from initial data loading and cleaning to model training, tuning, and evaluation.

***

### 📚 Dataset

The project uses the **B3DB (Blood-Brain Barrier DataBase)** dataset, which contains a large collection of compounds with experimentally measured permeability values.

* **Source:** The data is cloned from the official B3DB GitHub repository.

* **Key Data:** The analysis focuses on the `B3DB_classification_extended.tsv.gz` file, which includes over 1,600 molecular descriptors for each compound.

* **Target Variable:** The `BBB+/BBB-` column, which is mapped to a numerical label (**1** for BBB+ and **0** for BBB-) for classification.

***

### 🧪 Methodology

The following steps were taken to build and evaluate the predictive model:

#### 1. Data Cleaning & Preprocessing

* **Column Filtering**: Columns with more than 70% missing values were dropped to handle sparse data.

* **Low Variance Feature Removal**: Features with very low variance (below $10^{-3}$) were removed to eliminate noise and redundant information.

* **Missing Value Imputation**: The remaining missing values were imputed using a **MICE (IterativeImputer)** algorithm, which uses a model to predict and fill in missing data points.

* **Data Scaling**: The features were standardized using `StandardScaler` to ensure they have a mean of 0 and a standard deviation of 1, which improves model performance.

#### 2. Model Training & Tuning

* **Baseline Model**: A **Logistic Regression** model was trained first to establish a performance baseline, achieving an initial accuracy of approximately **88%** and a ROC AUC score of **0.937**.

* **Advanced Model**: An **XGBoost Classifier** was then used to build a more powerful model.

* **Hyperparameter Tuning**: To find the optimal model configuration, **Randomized Search Cross-Validation (`RandomizedSearchCV`)** was performed. The best-performing hyperparameters were identified for the final model.

#### 3. Feature Selection & Final Evaluation

* **Feature Importance**: The importance of each feature was extracted from the tuned XGBoost model. The top 20 most important features, such as `TopoPSA`, `nAcid`, and `nG12FARing`, were identified.

* **Final Model**: A final XGBoost model was trained on the top 100 most important features to reduce dimensionality and potentially improve generalization.

* **Cross-Validation Pipeline**: The entire preprocessing and modeling workflow was encapsulated in a `Pipeline` and evaluated using a **Stratified K-Fold** cross-validation strategy with 5 splits. This provided a reliable assessment of the model's performance on unseen data.

***

### ✅ Results

The final cross-validation results demonstrate a high-performing and stable model:

* **Mean Test Accuracy:** $0.8807 \\pm 0.0105$

* **Mean Test Precision:** $0.8847 \\pm 0.0082$

* **Mean Test Recall:** $0.9338 \\pm 0.0098$

* **Mean Test F1-Score:** $0.9086 \\pm 0.0081$

* **Mean Test ROC AUC:** $0.8612 \\pm 0.0113$

The difference between the train and test ROC AUC scores is small (`0.0629`), indicating that the model generalizes well and is not significantly overfitting the training data.

***

### 🛠️ Dependencies

To run this project, you will need the following Python libraries:

```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```

***

### 🚀 How to Run

1. Clone the repository to your local machine.

   ```
   git clone [https://github.com/kalashk/bbb-prediction.git](https://github.com/kalashk/bbb-prediction.git)
   cd bbb-prediction
   ```

2. Clone the B3DB dataset.

   ```
   git clone [https://github.com/theochem/B3DB](https://github.com/theochem/B3DB)
   ```

3. Ensure all dependencies are installed.

   ```
   pip install -r requirements.txt
   ```

   (Note: You may need to create a `requirements.txt` file from the list above.)

4. Run the main Python script containing the analysis code.

   ```
   python your_script_name.py
   

### in the name of Allah
# FraudDetection-Classification-ECommerce-Project
>>> Electronic Commerce - Winter 2025

**Attention: see result files (plots & csv files) in < ResultFiles > folder**

## TASK I: Exploratory Data Analysis (EDA) and Data Preprocessing

In this task, we are responsible for loading, inspecting, and cleaning the raw transaction dataset, followed by extracting key statistical insights. This includes the following steps:

1. **Load and inspect the dataset**: The dataset is read from `fraud_detection_2.csv`. Initial properties such as row count, column count, and column names are reported to confirm correct loading.

2. **Handle NULL values and duplicate rows**: All columns are checked for missing values. If any exist, the corresponding rows are dropped. Complete duplicate rows are identified and removed to ensure data integrity.

3. **Fraud count analysis per account**: Transactions are grouped by account ID and the total number of fraudulent records (`Class = 1`) per account is computed. The accounts with the highest and lowest (non-zero) fraud counts are identified.

4. **Amount statistics for fraudulent transactions**: All records labeled as fraud are filtered and descriptive statistics — mean, median, and standard deviation of the `Amount` column — are computed to characterize the financial profile of fraudulent activity.

5. **Balance analysis (last record per account)**: For each account, only the most recent transaction record is retained. The accounts with the highest and lowest transaction amounts among these final records are identified.

6. **Overall dataset summary and save cleaned data**: Key aggregate statistics are reported, including total unique accounts, total records, fraud vs. normal record counts, fraud percentage, and the class imbalance ratio (Normal/Fraud). The cleaned dataset is saved to `ResultFiles/1_cleaned_fraud_data.csv` for use in subsequent tasks.

--------------------------------------------------------------------------------------------------------------

## TASK II: Data Visualization and Exploratory Graphical Analysis

In this task, we focus on creating comprehensive visualizations to understand feature distributions, class separability, and fraud patterns across different transaction types. The main steps are as follows:

1. **Load cleaned data from Task I**  
   The preprocessed dataset saved in Task I (`1_cleaned_fraud_data.csv`) is loaded. Basic properties such as shape and column names are verified to ensure continuity between tasks.

2. **Histogram plots for feature distributions**  
   Histograms are generated for all anonymized transaction features (V1–V28) to visualize their individual distributions. Each histogram includes a kernel density estimate (KDE) overlay for smoother distribution representation. Additionally, the `Amount` feature is visualized both in linear and log10 scales to handle its wide range and skewness.

3. **Violin plots for class comparison**  
   To manage computational overhead and improve plot clarity, stratified sampling is applied (2000 samples per class). Violin plots are created for all V features and the `Amount` feature, showing distribution differences between normal (Class 0) and fraudulent (Class 1) transactions. These plots combine box plots and density curves to reveal central tendency, spread, and distributional shape differences across classes.

4. **Scatter plots for key features (V13, V15, V26)**  
   Individual scatter plots are created for V13, V15, and V26, color-coded by class, to visualize how these features separate fraud from normal transactions across the sample index. Additionally, pairwise scatter plots (V13 vs V15, V13 vs V26, V15 vs V26) are generated to explore potential two-dimensional relationships and clustering patterns between these features.

5. **Fraud analysis: Deposits vs Withdrawals**  
   For features V13, V15, and V26, transactions are categorized as deposits (positive values) or withdrawals (negative values). The number and percentage of fraudulent transactions in each category are calculated and compared using bar plots. This analysis reveals whether fraud is more prevalent in deposit or withdrawal activities, providing actionable insights into fraud behavior patterns.

All visualizations are saved to the `ResultFiles/Task2 Plots/` directory as high-resolution PNG files for inclusion in the final report and further analysis.

--------------------------------------------------------------------------------------------------------------

## TASK III: Fraud Detection Classification

In this task, we develop and compare machine learning models for fraud detection using the cleaned dataset from Task I and insights from Task II. The workflow includes feature engineering, feature selection, model training with class imbalance handling, and comprehensive evaluation. The main steps are as follows:

1. **Load cleaned data and assess class imbalance**  
   The preprocessed dataset (`1_cleaned_fraud_data.csv`) is loaded. Class distribution reveals a severe imbalance ratio of approximately 210:1 (normal vs fraud transactions), confirming the need for specialized handling techniques.

2. **Feature engineering**  
   Five new features are created to enhance model discriminative power:
   - `Amount_log`: Log transformation of transaction amount to handle right-skewed distribution
   - `is_withdrawal`: Binary indicator based on V15 < 0 (from Task II insights showing fraud prevalence in withdrawals)
   - `V15_V17`, `V15_V26`, `V17_V26`: Interaction features combining top discriminative variables identified in Task II

3. **Feature selection using Mutual Information (MI)**  
   Mutual Information scores are calculated for all features (original V1-V28 plus engineered features, excluding `Amount` to avoid data leakage). The top 20 features with highest MI scores are selected, including engineered features `V15_V17` (MI=0.0164, rank 4), `is_withdrawal` (MI=0.0154, rank 5), and `V17_V26` (MI=0.0116, rank 9). This statistical approach ensures features with strongest non-linear relationships to fraud are retained.

4. **Train-test split and feature scaling**  
   Data is split 80-20 with stratification to preserve class distribution in both sets (train: 79,494 samples; test: 19,874 samples). StandardScaler is applied to ensure features have mean=0 and std=1, critical for distance-based algorithms like Logistic Regression.

5. **Model training with class imbalance handling**  
   Two models are trained with `class_weight='balanced'` to automatically assign higher penalties to minority class (fraud) misclassifications:
   - **Logistic Regression**: Linear baseline model with C=0.1 regularization
   - **Random Forest**: Ensemble model with 100 trees, max_depth=10 to prevent overfitting

6. **Model evaluation with fraud-focused metrics**  
   Both models are evaluated using a **fixed threshold of 0.5** (no optimization) to ensure fair comparison. Metrics include:
   - **Precision**: Proportion of predicted frauds that are actual frauds
   - **Recall**: Proportion of actual frauds correctly detected (critical for fraud detection)
   - **F1-Score**: Harmonic mean of Precision and Recall
   - **F2-Score**: Weighted F-score with β=2, emphasizing Recall over Precision

7. **Results and visualization**  
   - **Confusion Matrices** (`10_confusion_matrices.png`): Visual comparison showing Random Forest achieves better balance between True Positives and False Positives
   - **Feature Importance** (`11_feature_importance.png`): Top 15 features from Random Forest reveal that engineered features (`V15_V17`, `is_withdrawal`) rank highly alongside original features (V17, V14, V12)
   - **Model Comparison** (`12_model_comparison.csv`): Summary table showing:
     * **Logistic Regression**: Precision=0.20, Recall=0.87, F1=0.33, F2=0.52
     * **Random Forest**: Precision=0.87, Recall=0.79, F1=0.83, F2=0.80

8. **Best model selection**  
   **Random Forest** is selected as the best model based on F1-Score (0.83), achieving superior balance between precision and recall. While Logistic Regression has higher recall (0.87), its extremely low precision (0.20) results in excessive false alarms, making it impractical for deployment.

All results, including confusion matrices, feature importance plot, and model comparison summary, are saved to `ResultFiles/Task3 Classification/` for inclusion in the final report.

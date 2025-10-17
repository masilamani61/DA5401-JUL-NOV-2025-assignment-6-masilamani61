#  Assignment 6 — Regression, Imputation, and Model Optimization Experiments

### Name - **V G Masilamani(DA25S005)**

##  Overview
This notebook implements and analyzes **multiple regression and imputation strategies** under controlled missingness, along with **regularized and classification model optimization**.

The workflow follows a structured research-style pipeline that begins with **data corruption (MAR mechanism)**, progresses through **various imputation models**, and concludes with **Ridge and Logistic Regression** tuning experiments.

---

##  Key Modules

### **1️ Data Preprocessing and MAR Missingness (5%)**
- Dataset: **UCI Credit Card Default Data**
- Introduced **Missing At Random (MAR)** missingness into 3 features:
  - `AGE`
  - `BILL_AMT1`
  - `PAY_AMT1`
- Missingness probability was conditioned on **`LIMIT_BAL` quantiles**.
- Ensured **exact 5% missing rate per selected variable**.
- Verified using a reproducible random seed (`RANDOM_STATE = 48`).

 **Outcome:**  
A realistic dataset with controlled missingness suitable for comparing imputation models.

---

### **2️ Distribution Analysis Before & After Missingness**
- Visualized histograms of affected features before and after missingness.
- Found that **data distribution shapes remained intact**, validating the MAR process.

 **Plots:** Distribution of `AGE`, `BILL_AMT1`, and `PAY_AMT1` before vs after MAR.

---

### **3️ Imputation Models Comparison**
Implemented and evaluated four approaches:
| Model | Description | Type |
|--------|--------------|------|
| **A** | Mean Imputation | Baseline |
| **B** | Linear Regression Imputation | Deterministic |
| **C** | Non-linear Regression (Random Forest) Imputation | Non-linear |
| **D** | Listwise Deletion | Data Removal |

Each imputed dataset was used to train a regression model, and the **R² and RMSE** values were compared.

 **Findings:**
- Imputation (Models A–C) retained more data and reduced bias.
- **Listwise Deletion (Model D)** performed poorly due to loss of samples.
- **Non-linear regression imputation** outperformed linear under non-linear relationships.

---

### **4️ Linear vs. Non-linear Regression Efficacy**
- Linear Regression captured linear dependencies well.
- Non-linear models (like Random Forest) performed better when the relationship between predictors and target deviated from linearity.

 **Conclusion:**  
Non-linear regression achieved higher accuracy due to flexible function approximation capability.

---

### **5️ Ridge Regression (wᴿ) vs. Maximum Likelihood (wᴹᴸ)**
Implemented **Ridge Regression** using:
- **Closed-form solution:**  
  \[
  w_R = (X^TX + \lambda I)^{-1} X^Ty
  \]
- **Gradient Descent Optimization:**  
  Updated weights iteratively using:
  \[
  w := w - \eta \left[ \frac{1}{n}X^T(Xw - y) + \lambda w \right]
  \]

#### Cross-Validation:
- Tuned λ (regularization parameter) across multiple values.
- Plotted validation error vs λ.
- Chose optimal λ minimizing validation loss.

#### Comparison:
| Model | Description | Test Error | Conclusion |
|--------|--------------|------------|-------------|
| **wᴹᴸ** | Ordinary Least Squares | Higher | Overfits |
| **wᴿ** | Ridge Regression | Lower | Better generalization |

 **Ridge Regression performed better** because it controls model complexity by penalizing large coefficients, reducing overfitting.

---

### **6️ Logistic Regression Fine-Tuning**
The final section performs **Logistic Regression tuning** to improve classification on an imbalanced dataset.

#### Process:
- Used **GridSearchCV** for hyperparameter tuning.
- Optimized for **F1-score of the minority class** using a custom scorer.
- Cross-validated across combinations of:
  - Regularization (`C`)
  - Penalty (`l1`, `l2`)
  - Solver (`liblinear`, `saga`)

#### Evaluation:
- Reported best parameters and F1-scores.
- Generated **confusion matrix** and **classification report**.
- Visualized F1-score trend across λ.

 **Insight:**
Using **F1 (minority)** instead of accuracy gives a balanced evaluation on imbalanced data, improving fairness and recall.

---

##  Experiments Summary

| Section | Objective | Outcome |
|----------|------------|----------|
| MAR Missingness | Simulate realistic missing data | Controlled 5% missing values |
| Imputation | Recover missing data | Non-linear > Linear > Mean |
| Listwise Deletion | Benchmark | Poor performance due to sample loss |
| Ridge Regression | Regularization | Reduced overfitting |
| Logistic Regression | Fine-tuning | Improved F1 (minority) via CV |

---

##  Implementation Environment
**Languages & Libraries:**
- Python 3.10+
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

**Hardware:**  
Standard CPU runtime; no GPU required.

---

##  How to Run
```bash
# Step 1: Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn

# Step 2: Open notebook
jupyter notebook Main.ipynb

# Step 3: Run all cells in order

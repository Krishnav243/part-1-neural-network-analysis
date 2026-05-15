# Part 1: Neural Network Fundamentals and Training Behavior Analysis

## Project Overview
This project focuses on building and analyzing a feed-forward neural network for customer churn prediction using a structured dataset. The objective is to understand neural network fundamentals including forward propagation, activation functions, loss calculation, backpropagation, and parameter updates.

The model predicts whether a customer is likely to churn:

- **0** → Customer retained  
- **1** → Customer churned

---

## Dataset Information

**Dataset Used:** `customer_churn_nn.csv`

### Dataset Description
The dataset contains customer-related information used to predict customer churn.

- **Rows:** 2000  
- **Columns:** 17

### Feature Types

#### Categorical Features
- region
- plan_type
- contract_type
- payment_method

#### Numerical Features
- tenure_months
- monthly_charges_inr
- avg_login_days_per_month
- support_tickets_last_90_days
- payment_delay_days
- data_usage_gb
- satisfaction_score
- last_complaint_days_ago
- discount_percent
- autopay_enabled
- referral_count

### Target Variable
`churn`
- **0** = retained customer
- **1** = churned customer

### Missing Value Check
No missing values were found in the dataset.

### Target Variable Distribution
The dataset is highly imbalanced:

- **98.45%** retained customers
- **1.55%** churned customers

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed `customer_id` column because it is only an identifier.
2. Applied **One-Hot Encoding** to categorical columns.
3. Used **StandardScaler** for feature scaling.
4. Split dataset into:
   - **80% Training Data**
   - **20% Testing Data**

---

## Neural Network Model

A feed-forward neural network was created using TensorFlow/Keras.

### Model Architecture
- Input Layer
- Hidden Layer 1 → 16 neurons (ReLU)
- Hidden Layer 2 → 8 neurons (ReLU)
- Output Layer → 1 neuron (Sigmoid)

### Model Configuration
- **Optimizer:** Adam
- **Loss Function:** Binary Crossentropy
- **Metric:** Accuracy
- **Epochs:** 20
- **Batch Size:** 32

---

## Model Performance

### Test Results
- **Test Accuracy:** 98.25%
- **Test Loss:** 0.0832

### Evaluation
The model achieved high accuracy; however, due to dataset imbalance, the model mostly predicted the majority class (non-churn customers).

Confusion matrix and evaluation outputs are available in the `results/` folder.

---

## Hyperparameter Experimentation

Three experiments were performed by changing:
- Number of neurons
- Learning rate
- Epochs
- Batch size

Results are available in:

```text
results/model_comparison_table.csv

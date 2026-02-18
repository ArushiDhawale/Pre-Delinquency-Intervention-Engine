# Pre-Delinquency Intervention Engine

### *AI-Powered Early Warning System for Loan Default Prevention*

## Project Overview

The **Pre-Delinquency Intervention Engine** is a Machine Learning solution designed to predict the likelihood of loan default *before* a missed payment occurs. Unlike traditional credit scoring that relies on static history, this engine analyzes **dynamic behavioral patterns**—such as sudden increases in quick-loan app usage, drops in discretionary spending, and irregular salary credits—to flag high-risk customers in real-time.

This project includes a **Random Forest Classifier** model and an interactive **Streamlit Dashboard** for loan officers to monitor risk and audit individual user profiles.

---

## Key Features

* **Behavioral Risk Analysis:** Detects financial distress signals (e.g., "Wallet Tightening" or "Debt Stacking" via UPI apps) weeks before default.
* **Live Scoring Engine:** Calculates risk probability (0-100%) in real-time based on transaction logs.
* **Interactive Dashboard:**
* **Overview:** Top 10 high-risk users requiring immediate intervention.
* **Manual Assessment:** Search any User ID to see their risk gauge and automated "AI Reasoning" text.
* **Model Audit:** Comparison of Random Forest vs. XGBoost/SVM performance.


* **Explainable AI:** Provides clear reasons for risk scores (e.g., *"Risk elevated due to 21+ visits to lending apps"*).

---

## Tech Stack

* **Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn (Random Forest, Logic Regression, SVM)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly Express, Plotly Graph Objects
* **Dashboard Framework:** Streamlit

---

## Project Structure

```bash
Pre-Delinquency-Engine/
│
├── dashboard.py                   # Main Streamlit Dashboard application
├── Pre Delinquency.ipynb          # Jupyter Notebook for EDA, Training & Model Selection
├── german_credit_data.csv         # Dataset 1: Demographic & Credit History
├── synthetic_transaction_data.csv # Dataset 2: Daily Transaction Logs (UPI, Salary, Spend)
└── README.md                      # Project Documentation

```

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/pre-delinquency-engine.git
cd pre-delinquency-engine

```

### 2. Install Dependencies

Ensure you have Python installed, then run:

```bash
pip install streamlit pandas numpy plotly scikit-learn

```

### 3. Place Data Files

Ensure the following two CSV files are in the **same directory** as `dashboard.py`:

* `german_credit_data.csv`
* `synthetic_transaction_data.csv`

### 4. Run the Dashboard

```bash
streamlit run dashboard.py

```

*The dashboard will open automatically in your default web browser at `http://localhost:8501`.*

---

## How It Works

The engine follows a 3-step pipeline to determine risk:

### Step 1: Feature Engineering (from Transaction Logs)

Instead of just looking at "Age" or "Job", the system calculates dynamic metrics from `synthetic_transaction_data.csv`:

* **UPI Lending Count:** How many times did the user transact with "UPI_Lending_App"? (High count = Distress).
* **Avg Discretionary Spend:** Is the user cutting back on lifestyle spending? (Low spend = Liquidity Crunch).
* **Avg Salary Day:** Is the salary arriving later than usual? (Irregularity = Instability).

### Step 2: Data Merging

These behavioral features are merged with the static `german_credit_data.csv` (Credit Amount, Duration, Housing status) to create a comprehensive **360-degree user profile**.

### Step 3: Risk Scoring (Random Forest Logic)

The model applies weighted logic derived from training:

* **High Risk:** High UPI Usage + Low Discretionary Spend + High Credit Amount.
* **Low Risk:** Consistent Salary + Stable Spending + Zero Lending App usage.

---

## Model Performance

After testing multiple algorithms (Logistic Regression, SVM, XGBoost), **Random Forest** was selected as the champion model.

| Metric | Random Forest Score | Why it matters? |
| --- | --- | --- |
| **Recall** | **87.9%** | Critical for risk; ensures we don't miss potential defaulters. |
| **ROC-AUC** | **0.935** | Indicates excellent separation between "Good" and "Bad" users. |
| **Precision** | **96.7%** | Ensures we don't falsely harass safe customers. |

---

## Dashboard Previews

### 1. Manual Risk Assessment
*Enter a User ID to see a speedometer gauge of their risk and a text explanation of **why** they are risky.*

### 2. Live Risk Overview
*A real-time leaderboard of the users most likely to default in the next 30 days.*

---

## Project by:
Arushi Dhawale, Methika M, Swarali Marwadi

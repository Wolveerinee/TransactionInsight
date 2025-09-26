# 🏦 Banking Transaction Insights Dashboard

A **Streamlit-based banking analytics platform** that simulates,
analyzes, and visualizes financial transaction data. The system provides
real-time fraud detection, customer segmentation, churn risk analysis,
and interactive dashboards for decision-making.

## 🚀 Features

### 📊 Core Modules

-   **Dashboard (`dashboard.py`)**
    -   Key performance indicators (revenue, active customers, avg.
        transaction, fraud rate).\
    -   Visualizations: Monthly revenue, transaction volume, category
        spending, customer segments, fraud distribution.
-   **Customer Analytics (`analytics.py`)**
    -   Customer segmentation using **K-Means clustering**.\
    -   Spending behavior and category preferences.\
    -   Churn risk prediction with multi-factor scoring.\
    -   Actionable insights and recommendations (high-value customers,
        churn prevention).
-   **Main Application (`app.py`)**
    -   Streamlit-powered navigation with multiple modules:
        -   **Dashboard** -- High-level transaction and customer KPIs.\
        -   **Real-Time Stream** -- Monitor live transactions with fraud
            alerts.\
        -   **Data Generation** -- Create synthetic customer &
            transaction datasets.\
        -   **Transaction Analysis** -- Category, temporal, and
            trend-based analysis.\
        -   **Fraud Detection** -- Anomaly-based fraud scoring and
            flagged transactions.\
        -   **Customer Analytics** -- Segmentation, churn risk, and
            retention analysis.\
        -   **Data Quality** -- Missing values, duplicates, type
            validation, and anomaly detection.

## 🛠️ Tech Stack

-   **Frontend:** [Streamlit](https://streamlit.io/)\
-   **Data Analysis:** Pandas, NumPy, Scikit-learn\
-   **Visualization:** Plotly (interactive charts & KPIs)\
-   **Database Support:** Custom DB manager integration\
-   **Machine Learning:** K-Means clustering, heuristic churn risk
    scoring\
-   **Real-time Processing:** Streaming transaction simulation &
    monitoring

## 📂 Project Structure

    ├── app.py              # Main Streamlit application with navigation
    ├── analytics.py        # Customer analytics, churn risk, segmentation
    ├── dashboard.py        # Dashboard visualizations and KPIs
    ├── fraud_detection.py  # Fraud detection logic (not uploaded here)
    ├── data_generator.py   # Synthetic data generation for customers & transactions
    ├── database.py         # Database manager for persistence
    ├── streaming.py        # Real-time streaming and dashboard
    ├── utils.py            # Helper functions & data quality checks

## 📸 Sample Visualizations

-   Customer distribution by spending & churn risk.\
-   Fraud detection heatmaps and flagged transactions.\
-   Monthly revenue and transaction activity trends.

## ⚡ How to Run

1.  Clone the repository

    ``` bash
    git clone https://github.com/your-username/banking-analytics-dashboard.git
    cd banking-analytics-dashboard
    ```

2.  Install dependencies

    ``` bash
    pip install -r requirements.txt
    ```

3.  Run the Streamlit app

    ``` bash
    streamlit run app.py
    ```

4.  Open in browser: <http://localhost:8501>

## 🎯 Use Cases

-   Banking & FinTech analytics.\
-   Fraud detection research and prototyping.\
-   Customer retention & churn analysis.\
-   Teaching tool for transaction-based data science.

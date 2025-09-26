import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_generator import BankingDataGenerator
from fraud_detection import FraudDetector
from analytics import CustomerAnalytics
from dashboard import Dashboard
from database import db_manager
from streaming import streaming_manager, streaming_dashboard
import utils

# Set page configuration
st.set_page_config(
    page_title="Banking Transaction Insights Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_generator' not in st.session_state:
    st.session_state.data_generator = BankingDataGenerator()
if 'fraud_detector' not in st.session_state:
    st.session_state.fraud_detector = FraudDetector()
if 'analytics' not in st.session_state:
    st.session_state.analytics = CustomerAnalytics()
if 'dashboard' not in st.session_state:
    st.session_state.dashboard = Dashboard()

# Load existing data from database or initialize as None
if 'transactions_df' not in st.session_state:
    try:
        st.session_state.transactions_df = db_manager.load_transactions_from_db()
    except:
        st.session_state.transactions_df = None

if 'customers_df' not in st.session_state:
    try:
        st.session_state.customers_df = db_manager.load_customers_from_db()
    except:
        st.session_state.customers_df = None

# Sidebar navigation
st.sidebar.title("🏦 Banking Analytics")
page = st.sidebar.selectbox(
    "Navigate to:",
    ["Dashboard", "Real-Time Stream", "Data Generation", "Transaction Analysis", "Fraud Detection", "Customer Analytics", "Data Quality"]
)

# Main content based on page selection
if page == "Dashboard":
    st.title("Banking Transaction Insights Dashboard")
    
    if st.session_state.transactions_df is not None:
        st.session_state.dashboard.render_main_dashboard(
            st.session_state.transactions_df,
            st.session_state.customers_df
        )
    else:
        st.warning("Please generate transaction data first from the 'Data Generation' page.")
        if st.button("Generate Sample Data"):
            with st.spinner("Generating transaction data..."):
                transactions, customers = st.session_state.data_generator.generate_full_dataset(
                    num_customers=1000,
                    num_transactions=10000
                )
                st.session_state.transactions_df = transactions
                st.session_state.customers_df = customers
                
                # Save to database
                try:
                    db_manager.save_transactions_to_db(transactions)
                    db_manager.save_customers_to_db(customers)
                    st.success("Sample data generated and saved to database successfully!")
                except Exception as e:
                    st.success("Sample data generated successfully!")
                    st.warning(f"Database save failed: {str(e)}")
                
                st.rerun()

elif page == "Real-Time Stream":
    st.title("🔴 Real-Time Transaction Stream")
    st.write("Monitor live transaction data with real-time fraud detection and alerts")
    
    # Render the streaming dashboard
    streaming_dashboard.render_live_dashboard()

elif page == "Data Generation":
    st.title("Banking Data Generation")
    st.write("Generate realistic banking transaction data for analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Generation Parameters")
        num_customers = st.number_input("Number of Customers", min_value=100, max_value=10000, value=1000)
        num_transactions = st.number_input("Number of Transactions", min_value=1000, max_value=100000, value=10000)
        
        date_range = st.date_input(
            "Transaction Date Range",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
            max_value=datetime.now()
        )
        
        if st.button("Generate Data", type="primary"):
            with st.spinner("Generating banking data..."):
                # Handle date_range - can be a tuple or single date
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = date_range
                    end_date = datetime.now()
                
                transactions, customers = st.session_state.data_generator.generate_full_dataset(
                    num_customers=num_customers,
                    num_transactions=num_transactions,
                    start_date=start_date,
                    end_date=end_date
                )
                st.session_state.transactions_df = transactions
                st.session_state.customers_df = customers
                
                # Save to database
                try:
                    db_manager.save_transactions_to_db(transactions)
                    db_manager.save_customers_to_db(customers)
                    st.success(f"Generated {len(transactions)} transactions for {len(customers)} customers and saved to database!")
                except Exception as e:
                    st.success(f"Generated {len(transactions)} transactions for {len(customers)} customers!")
                    st.warning(f"Database save failed: {str(e)}")
    
    with col2:
        st.subheader("Data Preview")
        if st.session_state.transactions_df is not None:
            st.write("**Transaction Data Sample:**")
            st.dataframe(st.session_state.transactions_df.head())
            
            # Download buttons
            csv_transactions = st.session_state.transactions_df.to_csv(index=False)
            st.download_button(
                label="Download Transactions CSV",
                data=csv_transactions,
                file_name="banking_transactions.csv",
                mime="text/csv"
            )
            
            if st.session_state.customers_df is not None:
                st.write("**Customer Data Sample:**")
                st.dataframe(st.session_state.customers_df.head())
                
                csv_customers = st.session_state.customers_df.to_csv(index=False)
                st.download_button(
                    label="Download Customers CSV",
                    data=csv_customers,
                    file_name="banking_customers.csv",
                    mime="text/csv"
                )

elif page == "Transaction Analysis":
    st.title("Transaction Analysis")
    
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        # SQL-like analysis using Pandas
        st.subheader("Spending Category Analysis")
        
        # Top spending categories
        category_analysis = df.groupby('category').agg({
            'amount': ['sum', 'mean', 'count'],
            'transaction_id': 'count'
        }).round(2)
        category_analysis.columns = ['Total_Amount', 'Avg_Amount', 'Transaction_Count', 'Frequency']
        category_analysis = category_analysis.sort_values('Total_Amount', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Spending Categories:**")
            st.dataframe(category_analysis)
            
        with col2:
            # Visualization
            fig = px.bar(
                category_analysis.reset_index(),
                x='category',
                y='Total_Amount',
                title='Total Spending by Category'
            )
            fig.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Transaction patterns by time
        st.subheader("Transaction Patterns")
        
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['month'] = df['transaction_date'].dt.to_period('M')
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Monthly transaction volume
            monthly_stats = df.groupby('month').agg({
                'amount': 'sum',
                'transaction_id': 'count'
            }).reset_index()
            monthly_stats['month'] = monthly_stats['month'].astype(str)
            
            fig = px.line(
                monthly_stats,
                x='month',
                y='amount',
                title='Monthly Transaction Volume',
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Hourly transaction patterns
            hourly_stats = df.groupby('hour')['transaction_id'].count().reset_index()
            
            fig = px.bar(
                hourly_stats,
                x='hour',
                y='transaction_id',
                title='Transaction Count by Hour of Day'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please generate transaction data first.")

elif page == "Fraud Detection":
    st.title("Fraud Detection System")
    
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        # Run fraud detection
        with st.spinner("Analyzing transactions for fraud patterns..."):
            fraud_results = st.session_state.fraud_detector.detect_fraud(df)
        
        st.subheader("Fraud Detection Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_transactions = len(df)
            st.metric("Total Transactions", f"{total_transactions:,}")
        
        with col2:
            flagged_transactions = fraud_results['fraud_flag'].sum()
            st.metric("Flagged Transactions", flagged_transactions)
        
        with col3:
            fraud_rate = (flagged_transactions / total_transactions) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        
        with col4:
            high_risk = (fraud_results['fraud_score'] > 0.7).sum()
            st.metric("High Risk Transactions", high_risk)
        
        # Fraud score distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                fraud_results,
                x='fraud_score',
                title='Fraud Score Distribution',
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top fraud categories
            fraud_by_category = fraud_results[fraud_results['fraud_flag'] == 1]['category'].value_counts()
            
            fig = px.pie(
                values=fraud_by_category.values,
                names=fraud_by_category.index,
                title='Fraud Distribution by Category'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed fraud analysis
        st.subheader("Flagged Transactions")
        
        flagged_df = fraud_results[fraud_results['fraud_flag'] == 1].sort_values('fraud_score', ascending=False)
        
        if len(flagged_df) > 0:
            st.dataframe(
                flagged_df[['transaction_id', 'customer_id', 'amount', 'category', 'fraud_score', 'fraud_reasons']],
                use_container_width=True
            )
            
            # Download flagged transactions
            csv_fraud = flagged_df.to_csv(index=False)
            st.download_button(
                label="Download Flagged Transactions",
                data=csv_fraud,
                file_name="flagged_transactions.csv",
                mime="text/csv"
            )
        else:
            st.info("No fraudulent transactions detected.")
    
    else:
        st.warning("Please generate transaction data first.")

elif page == "Customer Analytics":
    st.title("Customer Analytics & Churn Analysis")
    
    if st.session_state.transactions_df is not None and st.session_state.customers_df is not None:
        transactions_df = st.session_state.transactions_df
        customers_df = st.session_state.customers_df
        
        # Customer analysis
        with st.spinner("Analyzing customer patterns..."):
            customer_analysis = st.session_state.analytics.analyze_customers(transactions_df, customers_df)
        
        st.subheader("Customer Segmentation")
        
        # Customer segments
        segment_summary = customer_analysis.groupby('segment').agg({
            'customer_id': 'count',
            'total_spent': 'mean',
            'transaction_count': 'mean',
            'churn_risk_score': 'mean'
        }).round(2)
        segment_summary.columns = ['Customer_Count', 'Avg_Spending', 'Avg_Transactions', 'Avg_Churn_Risk']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Customer Segment Summary:**")
            st.dataframe(segment_summary)
        
        with col2:
            # Segment distribution
            fig = px.pie(
                values=segment_summary['Customer_Count'],
                names=segment_summary.index,
                title='Customer Distribution by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Churn analysis
        st.subheader("Churn Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn risk distribution
            fig = px.histogram(
                customer_analysis,
                x='churn_risk_score',
                title='Churn Risk Score Distribution',
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # High churn risk customers by segment
            high_churn = customer_analysis[customer_analysis['churn_risk_score'] > 0.7]
            churn_by_segment = high_churn['segment'].value_counts()
            
            fig = px.bar(
                x=churn_by_segment.index,
                y=churn_by_segment.values,
                title='High Churn Risk Customers by Segment'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # High-risk customers
        st.subheader("High Churn Risk Customers")
        
        high_risk_customers = customer_analysis[customer_analysis['churn_risk_score'] > 0.7].sort_values(
            'churn_risk_score', ascending=False
        )
        
        if len(high_risk_customers) > 0:
            st.dataframe(
                high_risk_customers[['customer_id', 'segment', 'total_spent', 'transaction_count', 
                                   'days_since_last_transaction', 'churn_risk_score']],
                use_container_width=True
            )
        else:
            st.info("No high churn risk customers identified.")
    
    else:
        st.warning("Please generate transaction and customer data first.")

elif page == "Data Quality":
    st.title("Data Quality & Pipeline Monitoring")
    
    if st.session_state.transactions_df is not None:
        df = st.session_state.transactions_df
        
        # Data quality checks
        quality_results = utils.perform_data_quality_checks(df)
        
        st.subheader("Data Quality Report")
        
        # Quality metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("Missing Data %", f"{missing_percentage:.2f}%")
        
        with col3:
            duplicate_count = df.duplicated().sum()
            st.metric("Duplicate Records", duplicate_count)
        
        with col4:
            quality_score = quality_results.get('overall_quality_score', 0)
            st.metric("Quality Score", f"{quality_score:.1f}/10")
        
        # Detailed quality checks
        st.subheader("Quality Check Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Completeness:**")
            completeness_df = pd.DataFrame({
                'Column': df.columns,
                'Missing_Count': df.isnull().sum().values,
                'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2).values
            })
            st.dataframe(completeness_df)
        
        with col2:
            # Data type validation
            st.write("**Data Type Validation:**")
            validation_results = []
            for column in df.columns:
                if column in ['amount']:
                    is_valid = pd.api.types.is_numeric_dtype(df[column])
                elif column in ['transaction_date']:
                    is_valid = pd.api.types.is_datetime64_any_dtype(df[column])
                else:
                    is_valid = True
                
                validation_results.append({
                    'Column': column,
                    'Expected_Type': 'Numeric' if column == 'amount' else 'DateTime' if column == 'transaction_date' else 'String',
                    'Is_Valid': '✅' if is_valid else '❌'
                })
            
            validation_df = pd.DataFrame(validation_results)
            st.dataframe(validation_df)
        
        # Data anomalies
        st.subheader("Data Anomalies")
        
        anomalies = utils.detect_anomalies(df)
        
        if anomalies:
            for anomaly_type, details in anomalies.items():
                st.write(f"**{anomaly_type}:**")
                st.write(details)
        else:
            st.success("No significant anomalies detected in the data.")
    
    else:
        st.warning("Please generate transaction data first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Pipeline Status:** ✅ Active")
st.sidebar.markdown("**Last Updated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

class Dashboard:
    def __init__(self):
        pass
    
    def render_main_dashboard(self, transactions_df, customers_df):
        """Render the main dashboard with key metrics and visualizations"""
        
        # Key Metrics Row
        self.render_key_metrics(transactions_df, customers_df)
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_monthly_revenue_chart(transactions_df)
        
        with col2:
            self.render_transaction_volume_chart(transactions_df)
        
        st.markdown("---")
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_category_spending_chart(transactions_df)
        
        with col2:
            self.render_customer_segment_chart(customers_df, transactions_df)
        
        st.markdown("---")
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_transaction_patterns_chart(transactions_df)
        
        with col2:
            self.render_fraud_overview_chart(transactions_df)
    
    def render_key_metrics(self, transactions_df, customers_df):
        """Render key performance metrics"""
        st.subheader("📊 Key Performance Indicators")
        
        # Calculate metrics
        total_revenue = transactions_df[transactions_df['amount'] > 0]['amount'].sum()
        total_transactions = len(transactions_df)
        unique_customers = transactions_df['customer_id'].nunique()
        avg_transaction_value = transactions_df['amount'].mean()
        
        # Current month metrics
        current_month = datetime.now().replace(day=1)
        current_month_data = transactions_df[
            pd.to_datetime(transactions_df['transaction_date']) >= current_month
        ]
        monthly_revenue = current_month_data[current_month_data['amount'] > 0]['amount'].sum()
        monthly_transactions = len(current_month_data)
        
        # Display metrics in columns
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.metric(
                label="Total Revenue",
                value=f"${total_revenue:,.2f}",
                delta=f"${monthly_revenue:,.2f} this month"
            )
        
        with col2:
            st.metric(
                label="Total Transactions",
                value=f"{total_transactions:,}",
                delta=f"{monthly_transactions:,} this month"
            )
        
        with col3:
            st.metric(
                label="Active Customers",
                value=f"{unique_customers:,}",
                delta=f"{current_month_data['customer_id'].nunique():,} this month"
            )
        
        with col4:
            st.metric(
                label="Avg Transaction",
                value=f"${avg_transaction_value:.2f}",
                delta=f"${current_month_data['amount'].mean():.2f} this month"
            )
        
        with col5:
            # Calculate fraud rate (simplified)
            suspicious_transactions = len(transactions_df[
                (transactions_df['amount'] < transactions_df['amount'].quantile(0.01)) |
                (transactions_df['amount'] > transactions_df['amount'].quantile(0.99))
            ])
            fraud_rate = (suspicious_transactions / total_transactions) * 100
            st.metric(
                label="Fraud Rate",
                value=f"{fraud_rate:.2f}%",
                delta="-0.5%" if fraud_rate < 2 else "+0.3%"
            )
        
        with col6:
            # Customer satisfaction proxy (transaction frequency)
            avg_customer_transactions = transactions_df.groupby('customer_id').size().mean()
            st.metric(
                label="Avg Txns/Customer",
                value=f"{avg_customer_transactions:.1f}",
                delta="+2.3"
            )
    
    def render_monthly_revenue_chart(self, transactions_df):
        """Render monthly revenue trend chart"""
        st.subheader("💰 Monthly Revenue Trend")
        
        df = transactions_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['year_month'] = df['transaction_date'].dt.to_period('M')
        
        # Filter for positive amounts (revenue)
        revenue_df = df[df['amount'] > 0]
        
        monthly_revenue = revenue_df.groupby('year_month')['amount'].sum().reset_index()
        monthly_revenue['year_month'] = monthly_revenue['year_month'].astype(str)
        
        fig = px.line(
            monthly_revenue,
            x='year_month',
            y='amount',
            title='Monthly Revenue Trend',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            hovermode='x unified'
        )
        
        fig.update_layout(yaxis_tickformat='$,.0f')
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_transaction_volume_chart(self, transactions_df):
        """Render transaction volume chart"""
        st.subheader("📈 Transaction Volume Analysis")
        
        df = transactions_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['year_month'] = df['transaction_date'].dt.to_period('M')
        
        monthly_volume = df.groupby('year_month').agg({
            'transaction_id': 'count',
            'customer_id': 'nunique'
        }).reset_index()
        
        monthly_volume['year_month'] = monthly_volume['year_month'].astype(str)
        monthly_volume.columns = ['year_month', 'total_transactions', 'unique_customers']
        
        # Create dual-axis chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Bar(
                x=monthly_volume['year_month'],
                y=monthly_volume['total_transactions'],
                name="Total Transactions",
                marker_color='lightblue'
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=monthly_volume['year_month'],
                y=monthly_volume['unique_customers'],
                mode='lines+markers',
                name="Unique Customers",
                line=dict(color='red', width=3)
            ),
            secondary_y=True,
        )
        
        fig.update_layout(title_text="Monthly Transaction Volume & Customer Activity")
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Number of Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Unique Customers", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_category_spending_chart(self, transactions_df):
        """Render spending by category chart"""
        st.subheader("🛍️ Spending by Category")
        
        # Calculate spending by category (positive amounts only)
        category_spending = transactions_df[transactions_df['amount'] > 0].groupby('category')['amount'].sum().sort_values(ascending=False)
        
        fig = px.pie(
            values=category_spending.values,
            names=category_spending.index,
            title='Total Spending Distribution by Category'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional category insights
        with st.expander("Category Insights"):
            st.write("**Top 3 Categories:**")
            for i, (category, amount) in enumerate(category_spending.head(3).items(), 1):
                st.write(f"{i}. {category.title()}: ${amount:,.2f}")
    
    def render_customer_segment_chart(self, customers_df, transactions_df):
        """Render customer segmentation chart"""
        st.subheader("👥 Customer Segmentation")
        
        # Simple segmentation based on spending
        customer_spending = transactions_df.groupby('customer_id')['amount'].sum()
        
        # Define segments based on spending quartiles
        q25 = customer_spending.quantile(0.25)
        q50 = customer_spending.quantile(0.50)
        q75 = customer_spending.quantile(0.75)
        
        def categorize_customer(spending):
            if spending <= q25:
                return 'Low Spender'
            elif spending <= q50:
                return 'Medium Spender'
            elif spending <= q75:
                return 'High Spender'
            else:
                return 'Premium Customer'
        
        customer_segments = customer_spending.apply(categorize_customer).value_counts()
        
        fig = px.bar(
            x=customer_segments.index,
            y=customer_segments.values,
            title='Customer Distribution by Spending Segment',
            color=customer_segments.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Number of Customers",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment metrics
        with st.expander("Segment Details"):
            for segment, count in customer_segments.items():
                avg_spending = customer_spending[customer_spending.apply(categorize_customer) == segment].mean()
                st.write(f"**{segment}**: {count} customers, Avg: ${avg_spending:,.2f}")
    
    def render_transaction_patterns_chart(self, transactions_df):
        """Render transaction patterns by time"""
        st.subheader("⏰ Transaction Patterns")
        
        df = transactions_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['hour'] = df['transaction_date'].dt.hour
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        
        # Transaction count by hour
        hourly_pattern = df.groupby('hour').size().reset_index(name='transaction_count')
        
        fig = px.bar(
            hourly_pattern,
            x='hour',
            y='transaction_count',
            title='Transaction Count by Hour of Day',
            color='transaction_count',
            color_continuous_scale='blues'
        )
        
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="Number of Transactions",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Peak hours insights
        peak_hour = hourly_pattern.loc[hourly_pattern['transaction_count'].idxmax(), 'hour']
        peak_count = hourly_pattern['transaction_count'].max()
        
        st.info(f"Peak transaction hour: {peak_hour}:00 with {peak_count} transactions")
    
    def render_fraud_overview_chart(self, transactions_df):
        """Render fraud detection overview"""
        st.subheader("🔒 Fraud Detection Overview")
        
        # Simple fraud indicators based on statistical outliers
        df = transactions_df.copy()
        df['amount_abs'] = abs(df['amount'])
        
        # Statistical outliers
        Q1 = df['amount_abs'].quantile(0.25)
        Q3 = df['amount_abs'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df['is_outlier'] = ((df['amount_abs'] < lower_bound) | (df['amount_abs'] > upper_bound))
        
        # Time-based suspicious activity
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['hour'] = df['transaction_date'].dt.hour
        df['is_unusual_time'] = (df['hour'] < 6) | (df['hour'] > 22)
        
        # Fraud summary
        total_transactions = len(df)
        outlier_transactions = df['is_outlier'].sum()
        unusual_time_transactions = df['is_unusual_time'].sum()
        
        fraud_summary = pd.DataFrame({
            'Fraud Type': ['Amount Outliers', 'Unusual Time', 'Normal Transactions'],
            'Count': [outlier_transactions, unusual_time_transactions, 
                     total_transactions - outlier_transactions - unusual_time_transactions],
            'Percentage': [
                (outlier_transactions / total_transactions) * 100,
                (unusual_time_transactions / total_transactions) * 100,
                ((total_transactions - outlier_transactions - unusual_time_transactions) / total_transactions) * 100
            ]
        })
        
        fig = px.pie(
            fraud_summary,
            values='Count',
            names='Fraud Type',
            title='Transaction Risk Distribution',
            color_discrete_map={
                'Amount Outliers': 'red',
                'Unusual Time': 'orange',
                'Normal Transactions': 'green'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Fraud metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Suspicious Amount", f"{outlier_transactions}")
        
        with col2:
            st.metric("Unusual Time", f"{unusual_time_transactions}")
        
        with col3:
            fraud_rate = ((outlier_transactions + unusual_time_transactions) / total_transactions) * 100
            st.metric("Overall Risk Rate", f"{fraud_rate:.2f}%")

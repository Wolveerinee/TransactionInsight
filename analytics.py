import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

class CustomerAnalytics:
    def __init__(self):
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42)
    
    def calculate_customer_metrics(self, transactions_df, customers_df):
        """Calculate comprehensive customer metrics"""
        # Transaction-based metrics
        customer_txn_metrics = transactions_df.groupby('customer_id').agg({
            'amount': ['sum', 'mean', 'std', 'count'],
            'transaction_date': ['min', 'max'],
            'category': lambda x: x.nunique()
        }).round(2)
        
        # Flatten column names
        customer_txn_metrics.columns = [
            'total_spent', 'avg_transaction_amount', 'transaction_amount_std', 
            'transaction_count', 'first_transaction', 'last_transaction', 'unique_categories'
        ]
        customer_txn_metrics = customer_txn_metrics.reset_index()
        
        # Calculate derived metrics
        current_date = datetime.now()
        customer_txn_metrics['days_since_first_transaction'] = (
            current_date - pd.to_datetime(customer_txn_metrics['first_transaction'])
        ).dt.days
        
        customer_txn_metrics['days_since_last_transaction'] = (
            current_date - pd.to_datetime(customer_txn_metrics['last_transaction'])
        ).dt.days
        
        customer_txn_metrics['transaction_frequency'] = (
            customer_txn_metrics['transaction_count'] / 
            (customer_txn_metrics['days_since_first_transaction'] + 1)
        ).round(3)
        
        # Monthly spending average
        customer_txn_metrics['monthly_avg_spending'] = (
            customer_txn_metrics['total_spent'] * 30 / 
            (customer_txn_metrics['days_since_first_transaction'] + 1)
        ).round(2)
        
        # Merge with customer demographics
        customer_metrics = customers_df.merge(customer_txn_metrics, on='customer_id', how='left')
        
        # Fill NaN values for customers with no transactions
        numeric_columns = customer_metrics.select_dtypes(include=[np.number]).columns
        customer_metrics[numeric_columns] = customer_metrics[numeric_columns].fillna(0)
        
        return customer_metrics
    
    def calculate_spending_patterns(self, transactions_df):
        """Analyze customer spending patterns by category"""
        # Spending by category per customer
        category_spending = transactions_df.groupby(['customer_id', 'category'])['amount'].sum().unstack(fill_value=0)
        
        # Calculate category preferences (percentage of total spending)
        category_preferences = category_spending.div(category_spending.sum(axis=1), axis=0).fillna(0)
        
        # Most frequent category per customer
        customer_top_category = transactions_df.groupby('customer_id')['category'].agg(lambda x: x.value_counts().index[0] if len(x) > 0 else 'none')
        
        return category_spending, category_preferences, customer_top_category
    
    def perform_customer_segmentation(self, customer_metrics):
        """Perform customer segmentation using K-means clustering"""
        # Select features for clustering
        clustering_features = [
            'total_spent', 'transaction_count', 'transaction_frequency',
            'monthly_avg_spending', 'unique_categories', 'days_since_last_transaction'
        ]
        
        # Prepare data for clustering
        clustering_data = customer_metrics[clustering_features].fillna(0)
        
        # Scale features
        scaled_data = self.scaler.fit_transform(clustering_data)
        
        # Perform clustering
        cluster_labels = self.kmeans.fit_predict(scaled_data)
        customer_metrics['cluster'] = cluster_labels
        
        # Assign meaningful segment names based on cluster characteristics
        segment_mapping = self._assign_segment_names(customer_metrics)
        customer_metrics['segment'] = customer_metrics['cluster'].map(segment_mapping)
        
        return customer_metrics
    
    def _assign_segment_names(self, customer_metrics):
        """Assign meaningful names to customer segments"""
        segment_stats = customer_metrics.groupby('cluster').agg({
            'total_spent': 'mean',
            'transaction_count': 'mean',
            'transaction_frequency': 'mean',
            'monthly_avg_spending': 'mean'
        })
        
        segment_mapping = {}
        
        for cluster in segment_stats.index:
            stats = segment_stats.loc[cluster]
            
            if stats['total_spent'] > segment_stats['total_spent'].median() * 1.5:
                if stats['transaction_frequency'] > segment_stats['transaction_frequency'].median():
                    segment_mapping[cluster] = 'High-Value Active'
                else:
                    segment_mapping[cluster] = 'High-Value Occasional'
            elif stats['transaction_frequency'] > segment_stats['transaction_frequency'].median() * 1.5:
                segment_mapping[cluster] = 'Frequent Low-Spender'
            else:
                segment_mapping[cluster] = 'Low-Activity'
        
        return segment_mapping
    
    def calculate_churn_risk(self, customer_metrics):
        """Calculate churn risk score for each customer"""
        # Factors that indicate churn risk
        # 1. Days since last transaction
        # 2. Decline in transaction frequency
        # 3. Decline in spending amount
        # 4. Reduced category diversity
        
        # Normalize factors to 0-1 scale
        customer_metrics['days_since_last_norm'] = (
            customer_metrics['days_since_last_transaction'] / 
            customer_metrics['days_since_last_transaction'].max()
        ).fillna(0)
        
        # Transaction frequency risk (lower frequency = higher risk)
        customer_metrics['freq_risk'] = 1 - (
            customer_metrics['transaction_frequency'] / 
            customer_metrics['transaction_frequency'].max()
        ).fillna(1)
        
        # Spending amount risk (lower spending = higher risk)
        customer_metrics['spending_risk'] = 1 - (
            customer_metrics['monthly_avg_spending'] / 
            customer_metrics['monthly_avg_spending'].max()
        ).fillna(1)
        
        # Category diversity risk (fewer categories = higher risk)
        customer_metrics['diversity_risk'] = 1 - (
            customer_metrics['unique_categories'] / 
            customer_metrics['unique_categories'].max()
        ).fillna(1)
        
        # Calculate weighted churn risk score
        customer_metrics['churn_risk_score'] = (
            customer_metrics['days_since_last_norm'] * 0.4 +
            customer_metrics['freq_risk'] * 0.3 +
            customer_metrics['spending_risk'] * 0.2 +
            customer_metrics['diversity_risk'] * 0.1
        )
        
        # Normalize to 0-1 range
        customer_metrics['churn_risk_score'] = np.clip(customer_metrics['churn_risk_score'], 0, 1)
        
        return customer_metrics
    
    def analyze_transaction_trends(self, transactions_df):
        """Analyze transaction trends over time"""
        df = transactions_df.copy()
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['year_month'] = df['transaction_date'].dt.to_period('M')
        
        # Monthly trends
        monthly_trends = df.groupby('year_month').agg({
            'amount': ['sum', 'mean', 'count'],
            'customer_id': 'nunique'
        }).round(2)
        
        monthly_trends.columns = ['total_amount', 'avg_amount', 'transaction_count', 'unique_customers']
        monthly_trends = monthly_trends.reset_index()
        monthly_trends['year_month'] = monthly_trends['year_month'].astype(str)
        
        # Category trends
        category_trends = df.groupby(['year_month', 'category'])['amount'].sum().unstack(fill_value=0)
        
        # Daily patterns
        df['day_of_week'] = df['transaction_date'].dt.day_name()
        df['hour'] = df['transaction_date'].dt.hour
        
        daily_patterns = df.groupby('day_of_week')['amount'].sum().reset_index()
        hourly_patterns = df.groupby('hour')['amount'].sum().reset_index()
        
        return {
            'monthly_trends': monthly_trends,
            'category_trends': category_trends,
            'daily_patterns': daily_patterns,
            'hourly_patterns': hourly_patterns
        }
    
    def generate_customer_insights(self, customer_metrics):
        """Generate actionable customer insights"""
        insights = {}
        
        # High churn risk customers
        high_churn_customers = customer_metrics[
            customer_metrics['churn_risk_score'] > 0.7
        ].sort_values('churn_risk_score', ascending=False)
        
        insights['high_churn_risk'] = {
            'count': len(high_churn_customers),
            'customers': high_churn_customers[['customer_id', 'segment', 'churn_risk_score']].to_dict('records')
        }
        
        # High-value customers
        high_value_customers = customer_metrics[
            customer_metrics['total_spent'] > customer_metrics['total_spent'].quantile(0.9)
        ].sort_values('total_spent', ascending=False)
        
        insights['high_value_customers'] = {
            'count': len(high_value_customers),
            'customers': high_value_customers[['customer_id', 'segment', 'total_spent']].to_dict('records')
        }
        
        # Segment analysis
        segment_summary = customer_metrics.groupby('segment').agg({
            'customer_id': 'count',
            'total_spent': ['mean', 'sum'],
            'churn_risk_score': 'mean',
            'transaction_frequency': 'mean'
        }).round(2)
        
        insights['segment_analysis'] = segment_summary.to_dict()
        
        # Recommendations
        insights['recommendations'] = self._generate_recommendations(customer_metrics)
        
        return insights
    
    def _generate_recommendations(self, customer_metrics):
        """Generate actionable recommendations based on customer analysis"""
        recommendations = []
        
        # Churn prevention
        high_churn_count = (customer_metrics['churn_risk_score'] > 0.7).sum()
        if high_churn_count > 0:
            recommendations.append(f"Immediate attention needed: {high_churn_count} customers at high churn risk")
        
        # Segment-specific recommendations
        segment_stats = customer_metrics.groupby('segment').agg({
            'churn_risk_score': 'mean',
            'total_spent': 'mean',
            'customer_id': 'count'
        })
        
        for segment, stats in segment_stats.iterrows():
            if stats['churn_risk_score'] > 0.5:
                recommendations.append(f"Focus retention efforts on {segment} segment ({stats['customer_id']} customers)")
            
            if segment == 'High-Value Active' and stats['churn_risk_score'] > 0.3:
                recommendations.append("Monitor high-value customers closely - showing early churn signals")
        
        # Growth opportunities
        low_activity_count = customer_metrics[customer_metrics['segment'] == 'Low-Activity']['customer_id'].count()
        if low_activity_count > 0:
            recommendations.append(f"Engagement opportunity: {low_activity_count} low-activity customers could be activated")
        
        return recommendations
    
    def analyze_customers(self, transactions_df, customers_df):
        """Main customer analytics pipeline"""
        # Calculate customer metrics
        customer_metrics = self.calculate_customer_metrics(transactions_df, customers_df)
        
        # Analyze spending patterns
        category_spending, category_preferences, top_categories = self.calculate_spending_patterns(transactions_df)
        
        # Perform customer segmentation
        customer_metrics = self.perform_customer_segmentation(customer_metrics)
        
        # Calculate churn risk
        customer_metrics = self.calculate_churn_risk(customer_metrics)
        
        # Add top category to customer metrics
        customer_metrics = customer_metrics.merge(
            top_categories.reset_index().rename(columns={'category': 'top_category'}),
            on='customer_id',
            how='left'
        )
        
        return customer_metrics

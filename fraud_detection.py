import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta

class FraudDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination='auto', random_state=42)
    
    def extract_features(self, df):
        """Extract features for fraud detection"""
        features_df = df.copy()
        
        # Convert transaction_date to datetime if not already
        features_df['transaction_date'] = pd.to_datetime(features_df['transaction_date'])
        
        # Amount-based features
        features_df['amount_abs'] = abs(features_df['amount'])
        features_df['is_large_amount'] = (features_df['amount_abs'] > features_df['amount_abs'].quantile(0.95)).astype(int)
        
        # Time-based features
        features_df['hour'] = features_df['transaction_date'].dt.hour
        features_df['day_of_week'] = features_df['transaction_date'].dt.dayofweek
        features_df['is_weekend'] = (features_df['day_of_week'] >= 5).astype(int)
        features_df['is_late_night'] = ((features_df['hour'] >= 22) | (features_df['hour'] <= 6)).astype(int)
        features_df['is_business_hours'] = ((features_df['hour'] >= 9) & (features_df['hour'] <= 17)).astype(int)
        
        # Customer behavior features
        customer_stats = features_df.groupby('customer_id').agg({
            'amount_abs': ['mean', 'std', 'count'],
            'transaction_date': ['min', 'max']
        }).round(2)
        
        customer_stats.columns = ['avg_amount', 'std_amount', 'transaction_count', 'first_transaction', 'last_transaction']
        customer_stats = customer_stats.reset_index()
        
        # Merge customer statistics
        features_df = features_df.merge(customer_stats, on='customer_id', how='left')
        
        # Deviation from normal behavior
        features_df['amount_deviation'] = abs(features_df['amount_abs'] - features_df['avg_amount']) / (features_df['std_amount'] + 1)
        
        # Frequency-based features
        features_df['days_since_first'] = (features_df['transaction_date'] - features_df['first_transaction']).dt.days
        features_df['days_since_last'] = (features_df['transaction_date'] - features_df['last_transaction']).dt.days
        
        # Velocity features (transactions per customer per day)
        daily_transactions = features_df.groupby(['customer_id', features_df['transaction_date'].dt.date]).size().reset_index(name='daily_txn_count')
        daily_transactions['transaction_date'] = pd.to_datetime(daily_transactions['transaction_date'])
        
        features_df = features_df.merge(
            daily_transactions,
            on=['customer_id', features_df['transaction_date'].dt.date],
            how='left'
        )
        
        # High velocity indicator
        features_df['is_high_velocity'] = (features_df['daily_txn_count'] > 10).astype(int)
        
        return features_df
    
    def detect_amount_anomalies(self, df):
        """Detect anomalies based on transaction amounts"""
        # Statistical outliers
        Q1 = df['amount_abs'].quantile(0.25)
        Q3 = df['amount_abs'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df['amount_outlier'] = ((df['amount_abs'] < lower_bound) | (df['amount_abs'] > upper_bound)).astype(int)
        
        # Z-score based detection
        df['amount_zscore'] = np.abs((df['amount_abs'] - df['amount_abs'].mean()) / df['amount_abs'].std())
        df['amount_zscore_outlier'] = (df['amount_zscore'] > 3).astype(int)
        
        return df
    
    def detect_time_anomalies(self, df):
        """Detect time-based anomalies"""
        # Unusual hours (very early morning)
        df['unusual_hour'] = ((df['hour'] >= 2) & (df['hour'] <= 5)).astype(int)
        
        # Rapid succession transactions
        df = df.sort_values(['customer_id', 'transaction_date'])
        df['time_diff'] = df.groupby('customer_id')['transaction_date'].diff().dt.total_seconds() / 60  # minutes
        df['rapid_succession'] = (df['time_diff'] < 5).astype(int)  # Less than 5 minutes apart
        
        return df
    
    def detect_pattern_anomalies(self, df):
        """Detect pattern-based anomalies using machine learning"""
        # Prepare features for ML model
        ml_features = [
            'amount_abs', 'hour', 'day_of_week', 'is_weekend', 'is_late_night',
            'amount_deviation', 'daily_txn_count', 'is_high_velocity'
        ]
        
        # Fill NaN values
        feature_data = df[ml_features].fillna(0)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(feature_data)
        
        # Isolation Forest for anomaly detection
        anomaly_scores = self.isolation_forest.fit_predict(scaled_features)
        df['isolation_anomaly'] = (anomaly_scores == -1).astype(int)
        
        # DBSCAN clustering to find outliers
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(scaled_features)
        df['cluster_outlier'] = (clusters == -1).astype(int)
        
        return df
    
    def calculate_fraud_score(self, df):
        """Calculate comprehensive fraud score"""
        # Define fraud indicators and their weights
        fraud_indicators = {
            'amount_outlier': 0.15,
            'amount_zscore_outlier': 0.15,
            'unusual_hour': 0.10,
            'rapid_succession': 0.20,
            'is_high_velocity': 0.15,
            'isolation_anomaly': 0.15,
            'cluster_outlier': 0.10
        }
        
        # Calculate weighted fraud score
        df['fraud_score'] = 0
        for indicator, weight in fraud_indicators.items():
            if indicator in df.columns:
                df['fraud_score'] += df[indicator] * weight
        
        # Additional scoring based on amount deviation
        df['fraud_score'] += np.clip(df['amount_deviation'] / 10, 0, 0.2)
        
        # Normalize fraud score to 0-1 range
        df['fraud_score'] = np.clip(df['fraud_score'], 0, 1)
        
        return df
    
    def generate_fraud_reasons(self, df):
        """Generate human-readable fraud reasons"""
        def get_fraud_reasons(row):
            reasons = []
            
            if row.get('amount_outlier', 0):
                reasons.append("Unusual transaction amount")
            if row.get('amount_zscore_outlier', 0):
                reasons.append("Statistical amount anomaly")
            if row.get('unusual_hour', 0):
                reasons.append("Transaction at unusual hour")
            if row.get('rapid_succession', 0):
                reasons.append("Rapid succession transactions")
            if row.get('is_high_velocity', 0):
                reasons.append("High transaction velocity")
            if row.get('isolation_anomaly', 0):
                reasons.append("Pattern anomaly detected")
            if row.get('cluster_outlier', 0):
                reasons.append("Behavioral outlier")
            if row.get('amount_deviation', 0) > 3:
                reasons.append("Significant deviation from normal spending")
            
            return "; ".join(reasons) if reasons else "Low risk transaction"
        
        df['fraud_reasons'] = df.apply(get_fraud_reasons, axis=1)
        return df
    
    def detect_fraud(self, transactions_df, fraud_threshold=0.5):
        """Main fraud detection pipeline"""
        df = transactions_df.copy()
        
        # Extract features
        df = self.extract_features(df)
        
        # Detect different types of anomalies
        df = self.detect_amount_anomalies(df)
        df = self.detect_time_anomalies(df)
        df = self.detect_pattern_anomalies(df)
        
        # Calculate fraud score
        df = self.calculate_fraud_score(df)
        
        # Generate fraud reasons
        df = self.generate_fraud_reasons(df)
        
        # Flag transactions as fraudulent
        df['fraud_flag'] = (df['fraud_score'] > fraud_threshold).astype(int)
        
        # Return relevant columns
        result_columns = [
            'transaction_id', 'customer_id', 'transaction_date', 'amount', 'category',
            'merchant', 'fraud_score', 'fraud_flag', 'fraud_reasons'
        ]
        
        return df[result_columns]
    
    def get_fraud_summary(self, fraud_results):
        """Generate fraud detection summary statistics"""
        total_transactions = len(fraud_results)
        flagged_transactions = fraud_results['fraud_flag'].sum()
        fraud_rate = (flagged_transactions / total_transactions) * 100 if total_transactions > 0 else 0
        
        avg_fraud_score = fraud_results['fraud_score'].mean()
        high_risk_count = (fraud_results['fraud_score'] > 0.7).sum()
        
        summary = {
            'total_transactions': total_transactions,
            'flagged_transactions': flagged_transactions,
            'fraud_rate_percent': round(fraud_rate, 2),
            'average_fraud_score': round(avg_fraud_score, 3),
            'high_risk_transactions': high_risk_count,
            'fraud_by_category': fraud_results[fraud_results['fraud_flag'] == 1]['category'].value_counts().to_dict()
        }
        
        return summary

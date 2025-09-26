import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def perform_data_quality_checks(df):
    """Perform comprehensive data quality checks on transaction data"""
    
    quality_results = {}
    
    # Basic data info
    quality_results['total_records'] = len(df)
    quality_results['total_columns'] = len(df.columns)
    
    # Missing data analysis
    missing_data = df.isnull().sum()
    quality_results['missing_data'] = missing_data.to_dict()
    quality_results['missing_percentage'] = (missing_data / len(df) * 100).round(2).to_dict()
    
    # Duplicate records
    quality_results['duplicate_records'] = df.duplicated().sum()
    quality_results['duplicate_percentage'] = (df.duplicated().sum() / len(df) * 100).round(2)
    
    # Data type validation
    quality_results['data_types'] = df.dtypes.to_dict()
    
    # Range validation for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    quality_results['numeric_ranges'] = {}
    
    for col in numeric_columns:
        quality_results['numeric_ranges'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'mean': df[col].mean(),
            'std': df[col].std()
        }
    
    # Date validation
    if 'transaction_date' in df.columns:
        df_temp = df.copy()
        df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'], errors='coerce')
        
        quality_results['date_validation'] = {
            'invalid_dates': df_temp['transaction_date'].isnull().sum(),
            'date_range': {
                'earliest': df_temp['transaction_date'].min(),
                'latest': df_temp['transaction_date'].max()
            }
        }
    
    # Business rule validation
    if 'amount' in df.columns:
        quality_results['amount_validation'] = {
            'zero_amounts': (df['amount'] == 0).sum(),
            'negative_amounts': (df['amount'] < 0).sum(),
            'positive_amounts': (df['amount'] > 0).sum(),
            'extreme_values': {
                'very_large': (abs(df['amount']) > df['amount'].abs().quantile(0.99)).sum(),
                'very_small': (abs(df['amount']) < 0.01).sum()
            }
        }
    
    # Consistency checks
    if 'customer_id' in df.columns:
        quality_results['customer_consistency'] = {
            'unique_customers': df['customer_id'].nunique(),
            'transactions_per_customer': {
                'min': df.groupby('customer_id').size().min(),
                'max': df.groupby('customer_id').size().max(),
                'mean': df.groupby('customer_id').size().mean()
            }
        }
    
    # Calculate overall quality score (0-10 scale)
    quality_score = calculate_quality_score(quality_results)
    quality_results['overall_quality_score'] = quality_score
    
    return quality_results

def calculate_quality_score(quality_results):
    """Calculate an overall data quality score"""
    score = 10.0
    
    # Deduct points for missing data
    avg_missing_percentage = np.mean(list(quality_results['missing_percentage'].values()))
    score -= (avg_missing_percentage / 10)  # Deduct up to 1 point for every 10% missing
    
    # Deduct points for duplicates
    duplicate_percentage = quality_results['duplicate_percentage']
    score -= (duplicate_percentage / 5)  # Deduct up to 2 points for duplicates
    
    # Deduct points for invalid dates
    if 'date_validation' in quality_results:
        invalid_date_percentage = (quality_results['date_validation']['invalid_dates'] / 
                                 quality_results['total_records']) * 100
        score -= (invalid_date_percentage / 10)
    
    return max(0, min(10, score))

def detect_anomalies(df):
    """Detect various types of anomalies in transaction data"""
    
    anomalies = {}
    
    # Amount-based anomalies
    if 'amount' in df.columns:
        # Statistical outliers using IQR method
        Q1 = df['amount'].abs().quantile(0.25)
        Q3 = df['amount'].abs().quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df['amount'].abs() < lower_bound) | (df['amount'].abs() > upper_bound)]
        
        if len(outliers) > 0:
            anomalies['Amount Outliers'] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100,
                'description': f"Transactions with amounts outside normal range ({lower_bound:.2f} - {upper_bound:.2f})"
            }
    
    # Time-based anomalies
    if 'transaction_date' in df.columns:
        df_temp = df.copy()
        df_temp['transaction_date'] = pd.to_datetime(df_temp['transaction_date'])
        df_temp['hour'] = df_temp['transaction_date'].dt.hour
        
        # Unusual hours (very early morning)
        unusual_hours = df_temp[(df_temp['hour'] >= 2) & (df_temp['hour'] <= 5)]
        
        if len(unusual_hours) > 0:
            anomalies['Unusual Time Transactions'] = {
                'count': len(unusual_hours),
                'percentage': (len(unusual_hours) / len(df)) * 100,
                'description': "Transactions occurring between 2 AM and 5 AM"
            }
    
    # Customer behavior anomalies
    if 'customer_id' in df.columns:
        customer_txn_counts = df.groupby('customer_id').size()
        
        # Customers with unusually high transaction counts
        high_volume_threshold = customer_txn_counts.quantile(0.95)
        high_volume_customers = customer_txn_counts[customer_txn_counts > high_volume_threshold]
        
        if len(high_volume_customers) > 0:
            anomalies['High Volume Customers'] = {
                'count': len(high_volume_customers),
                'percentage': (len(high_volume_customers) / df['customer_id'].nunique()) * 100,
                'description': f"Customers with more than {high_volume_threshold:.0f} transactions"
            }
    
    # Category distribution anomalies
    if 'category' in df.columns:
        category_counts = df['category'].value_counts()
        expected_min_percentage = 1.0  # Expected minimum 1% per category
        
        rare_categories = category_counts[category_counts < (len(df) * expected_min_percentage / 100)]
        
        if len(rare_categories) > 0:
            anomalies['Rare Categories'] = {
                'categories': rare_categories.index.tolist(),
                'description': f"Categories with less than {expected_min_percentage}% of total transactions"
            }
    
    return anomalies

def generate_data_summary(df):
    """Generate a comprehensive data summary"""
    
    summary = {}
    
    # Basic statistics
    summary['shape'] = df.shape
    summary['memory_usage'] = df.memory_usage(deep=True).sum()
    
    # Column types
    summary['column_types'] = {
        'numeric': df.select_dtypes(include=[np.number]).columns.tolist(),
        'datetime': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'categorical': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Unique value counts
    summary['unique_values'] = {}
    for col in df.columns:
        summary['unique_values'][col] = df[col].nunique()
    
    # Missing value summary
    summary['missing_values'] = df.isnull().sum().to_dict()
    
    # For numeric columns, provide statistical summary
    numeric_summary = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        numeric_summary[col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75)
        }
    
    summary['numeric_statistics'] = numeric_summary
    
    return summary

def validate_transaction_data(df):
    """Validate transaction data against business rules"""
    
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Required columns check
    required_columns = ['transaction_id', 'customer_id', 'amount', 'transaction_date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Data type validation
    if 'amount' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['amount']):
            validation_results['is_valid'] = False
            validation_results['errors'].append("Amount column must be numeric")
    
    if 'transaction_date' in df.columns:
        try:
            pd.to_datetime(df['transaction_date'])
        except:
            validation_results['warnings'].append("Some transaction dates may be invalid")
    
    # Business rule validation
    if 'amount' in df.columns:
        # Check for zero amounts
        zero_amounts = (df['amount'] == 0).sum()
        if zero_amounts > 0:
            validation_results['warnings'].append(f"Found {zero_amounts} transactions with zero amount")
        
        # Check for extreme amounts
        extreme_threshold = df['amount'].abs().quantile(0.999)
        extreme_amounts = (df['amount'].abs() > extreme_threshold).sum()
        if extreme_amounts > 0:
            validation_results['warnings'].append(f"Found {extreme_amounts} transactions with extreme amounts")
    
    # Duplicate transaction ID check
    if 'transaction_id' in df.columns:
        duplicate_ids = df['transaction_id'].duplicated().sum()
        if duplicate_ids > 0:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Found {duplicate_ids} duplicate transaction IDs")
    
    return validation_results

def clean_transaction_data(df):
    """Clean and prepare transaction data"""
    
    cleaned_df = df.copy()
    
    # Convert transaction_date to datetime
    if 'transaction_date' in cleaned_df.columns:
        cleaned_df['transaction_date'] = pd.to_datetime(cleaned_df['transaction_date'], errors='coerce')
    
    # Remove duplicate transactions
    if 'transaction_id' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop_duplicates(subset=['transaction_id'])
    
    # Handle missing values
    # For numeric columns, fill with median
    numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if cleaned_df[col].isnull().any():
            cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    # For categorical columns, fill with mode
    categorical_columns = cleaned_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        if cleaned_df[col].isnull().any():
            mode_value = cleaned_df[col].mode()
            if len(mode_value) > 0:
                cleaned_df[col] = cleaned_df[col].fillna(mode_value[0])
            else:
                cleaned_df[col] = cleaned_df[col].fillna('Unknown')
    
    # Remove rows with invalid dates
    if 'transaction_date' in cleaned_df.columns:
        cleaned_df = cleaned_df.dropna(subset=['transaction_date'])
    
    # Sort by transaction date
    if 'transaction_date' in cleaned_df.columns:
        cleaned_df = cleaned_df.sort_values('transaction_date')
    
    # Reset index
    cleaned_df = cleaned_df.reset_index(drop=True)
    
    return cleaned_df

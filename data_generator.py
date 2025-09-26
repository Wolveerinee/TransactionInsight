import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from faker import Faker

class BankingDataGenerator:
    def __init__(self):
        self.fake = Faker()
        Faker.seed(42)  # For reproducible results
        random.seed(42)
        np.random.seed(42)
        
        # Transaction categories with realistic spending patterns
        self.categories = {
            'groceries': {'weight': 0.25, 'avg_amount': 75, 'std': 25},
            'gas': {'weight': 0.15, 'avg_amount': 45, 'std': 15},
            'restaurants': {'weight': 0.20, 'avg_amount': 35, 'std': 20},
            'shopping': {'weight': 0.15, 'avg_amount': 120, 'std': 80},
            'utilities': {'weight': 0.08, 'avg_amount': 150, 'std': 50},
            'entertainment': {'weight': 0.07, 'avg_amount': 60, 'std': 30},
            'healthcare': {'weight': 0.05, 'avg_amount': 200, 'std': 150},
            'transport': {'weight': 0.03, 'avg_amount': 25, 'std': 10},
            'other': {'weight': 0.02, 'avg_amount': 100, 'std': 75}
        }
        
        # Transaction types
        self.transaction_types = ['debit', 'credit', 'withdrawal', 'deposit', 'transfer']
        
        # Merchant names by category
        self.merchants = {
            'groceries': ['Walmart', 'Target', 'Kroger', 'Safeway', 'Whole Foods'],
            'gas': ['Shell', 'BP', 'Exxon', 'Chevron', 'Mobil'],
            'restaurants': ['McDonald\'s', 'Starbucks', 'Subway', 'Pizza Hut', 'Local Diner'],
            'shopping': ['Amazon', 'Best Buy', 'Macy\'s', 'Nike', 'Apple Store'],
            'utilities': ['Electric Co', 'Gas Company', 'Water Dept', 'Internet Provider'],
            'entertainment': ['Netflix', 'Movie Theater', 'Spotify', 'Gaming Store'],
            'healthcare': ['Local Pharmacy', 'Medical Center', 'Dental Office'],
            'transport': ['Uber', 'Lyft', 'Bus Transit', 'Taxi Co'],
            'other': ['Online Service', 'Subscription', 'Misc Store']
        }
    
    def generate_customers(self, num_customers):
        """Generate realistic customer data"""
        customers = []
        
        for i in range(num_customers):
            # Customer demographics
            customer_id = f"CUST_{i+1:06d}"
            
            # Age distribution (18-80, normal distribution around 45)
            age = max(18, min(80, int(np.random.normal(45, 15))))
            
            # Income based on age and some randomness
            base_income = 30000 + (age - 18) * 1000 + np.random.normal(0, 15000)
            annual_income = max(20000, int(base_income))
            
            # Customer segment based on income and age
            if annual_income > 100000:
                segment = 'premium'
            elif annual_income > 60000:
                segment = 'standard'
            else:
                segment = 'basic'
            
            # Account opening date
            account_opened = self.fake.date_between(start_date='-5y', end_date='-1m')
            
            customers.append({
                'customer_id': customer_id,
                'age': age,
                'annual_income': annual_income,
                'segment': segment,
                'account_opened_date': account_opened,
                'city': self.fake.city(),
                'state': self.fake.state_abbr()
            })
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df, num_transactions, start_date=None, end_date=None):
        """Generate realistic transaction data"""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()
        
        transactions = []
        
        for i in range(num_transactions):
            # Select random customer
            customer = customers_df.sample(1).iloc[0]
            customer_id = customer['customer_id']
            
            # Generate transaction date
            transaction_date = self.fake.date_time_between(start_date=start_date, end_date=end_date)
            
            # Select category based on weights
            categories = list(self.categories.keys())
            weights = [self.categories[cat]['weight'] for cat in categories]
            category = np.random.choice(categories, p=weights)
            
            # Generate amount based on category and customer segment
            base_amount = self.categories[category]['avg_amount']
            std_amount = self.categories[category]['std']
            
            # Adjust amount based on customer segment
            segment_multiplier = {'basic': 0.7, 'standard': 1.0, 'premium': 1.5}
            multiplier = segment_multiplier.get(customer['segment'], 1.0)
            
            amount = max(1, abs(np.random.normal(base_amount * multiplier, std_amount)))
            
            # Determine transaction type
            if category in ['deposit', 'transfer']:
                transaction_type = category
            else:
                transaction_type = np.random.choice(['debit', 'credit'], p=[0.7, 0.3])
            
            # Adjust amount sign based on transaction type
            if transaction_type in ['withdrawal', 'debit']:
                amount = -abs(amount)
            
            # Select merchant
            merchant = np.random.choice(self.merchants[category])
            
            # Generate transaction ID
            transaction_id = f"TXN_{i+1:08d}"
            
            # Add some randomness to create realistic patterns
            # Weekend transactions might be different
            if transaction_date.weekday() >= 5:  # Weekend
                if category in ['restaurants', 'entertainment', 'shopping']:
                    amount *= np.random.uniform(1.1, 1.3)  # Higher weekend spending
            
            # Late night transactions (potential fraud indicators)
            hour = transaction_date.hour
            is_late_night = hour < 6 or hour > 22
            
            transactions.append({
                'transaction_id': transaction_id,
                'customer_id': customer_id,
                'transaction_date': transaction_date,
                'amount': round(amount, 2),
                'category': category,
                'merchant': merchant,
                'transaction_type': transaction_type,
                'is_weekend': transaction_date.weekday() >= 5,
                'is_late_night': is_late_night,
                'hour': hour
            })
        
        return pd.DataFrame(transactions)
    
    def inject_fraud_patterns(self, transactions_df, fraud_rate=0.02):
        """Inject realistic fraud patterns into transaction data"""
        num_fraud = int(len(transactions_df) * fraud_rate)
        fraud_indices = np.random.choice(transactions_df.index, size=num_fraud, replace=False)
        
        for idx in fraud_indices:
            # Different fraud patterns
            fraud_type = np.random.choice(['amount_spike', 'unusual_time', 'unusual_location', 'rapid_succession'])
            
            if fraud_type == 'amount_spike':
                # Unusually large amounts
                original_amount = abs(transactions_df.loc[idx, 'amount'])
                transactions_df.loc[idx, 'amount'] = -original_amount * np.random.uniform(5, 15)
            
            elif fraud_type == 'unusual_time':
                # Very late night transactions
                fraud_hour = np.random.choice([2, 3, 4])
                fraud_date = transactions_df.loc[idx, 'transaction_date'].replace(hour=fraud_hour)
                transactions_df.loc[idx, 'transaction_date'] = fraud_date
                transactions_df.loc[idx, 'is_late_night'] = True
                transactions_df.loc[idx, 'hour'] = fraud_hour
            
            elif fraud_type == 'rapid_succession':
                # Multiple transactions in short time
                base_date = transactions_df.loc[idx, 'transaction_date']
                for j in range(3):  # Create 3 rapid transactions
                    if idx + j < len(transactions_df):
                        new_date = base_date + timedelta(minutes=np.random.randint(1, 10))
                        transactions_df.loc[idx + j, 'transaction_date'] = new_date
        
        return transactions_df
    
    def generate_full_dataset(self, num_customers=1000, num_transactions=10000, start_date=None, end_date=None):
        """Generate complete banking dataset with customers and transactions"""
        # Generate customers
        customers_df = self.generate_customers(num_customers)
        
        # Generate transactions
        transactions_df = self.generate_transactions(customers_df, num_transactions, start_date, end_date)
        
        # Inject some fraud patterns
        transactions_df = self.inject_fraud_patterns(transactions_df)
        
        # Sort transactions by date
        transactions_df = transactions_df.sort_values('transaction_date').reset_index(drop=True)
        
        return transactions_df, customers_df

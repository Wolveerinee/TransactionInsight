import os
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, Float, DateTime, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection
DATABASE_URL = os.getenv('DATABASE_URL')

class DatabaseManager:
    def __init__(self):
        self.engine = None
        self.SessionLocal = None
        self.Base = declarative_base()
        self.connect()
    
    def connect(self):
        """Establish database connection"""
        try:
            if not DATABASE_URL:
                raise ValueError("DATABASE_URL environment variable not set")
            
            self.engine = create_engine(DATABASE_URL)
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            logger.info("Database connection established successfully")
            self.create_tables()
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            # Don't raise in production, allow app to work without database
            pass
    
    def create_tables(self):
        """Create all necessary tables"""
        if not self.engine:
            return
        
        try:
            # Create tables using raw SQL for better control
            with self.engine.connect() as conn:
                # User alerts table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS user_alerts (
                        id SERIAL PRIMARY KEY,
                        alert_name VARCHAR(255) NOT NULL,
                        alert_type VARCHAR(100) NOT NULL,
                        threshold_value FLOAT NOT NULL,
                        operator VARCHAR(10) NOT NULL,
                        is_active BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_triggered TIMESTAMP,
                        trigger_count INTEGER DEFAULT 0
                    )
                """))
                
                # Report history table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS report_history (
                        id SERIAL PRIMARY KEY,
                        report_name VARCHAR(255) NOT NULL,
                        report_type VARCHAR(100) NOT NULL,
                        file_path VARCHAR(500),
                        parameters JSON,
                        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        file_size INTEGER,
                        status VARCHAR(50) DEFAULT 'completed'
                    )
                """))
                
                # ML model metadata table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS ml_models (
                        id SERIAL PRIMARY KEY,
                        model_name VARCHAR(255) NOT NULL,
                        model_type VARCHAR(100) NOT NULL,
                        version VARCHAR(50) NOT NULL,
                        accuracy_score FLOAT,
                        precision_score FLOAT,
                        recall_score FLOAT,
                        f1_score FLOAT,
                        training_data_size INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT FALSE,
                        model_parameters JSON
                    )
                """))
                
                # Streaming data log table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS streaming_data_log (
                        id SERIAL PRIMARY KEY,
                        data_source VARCHAR(100) NOT NULL,
                        records_processed INTEGER NOT NULL,
                        processing_time_ms INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'success',
                        error_message TEXT
                    )
                """))
                
                # Export jobs table
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS export_jobs (
                        id SERIAL PRIMARY KEY,
                        job_name VARCHAR(255) NOT NULL,
                        export_type VARCHAR(100) NOT NULL,
                        export_format VARCHAR(50) NOT NULL,
                        file_path VARCHAR(500),
                        scheduled_time TIMESTAMP,
                        executed_at TIMESTAMP,
                        status VARCHAR(50) DEFAULT 'pending',
                        record_count INTEGER,
                        file_size INTEGER
                    )
                """))
                
                conn.commit()
                logger.info("All database tables created successfully")
                
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def save_transactions_to_db(self, transactions_df):
        """Save transaction data to database"""
        try:
            transactions_df.to_sql('transactions', self.engine, if_exists='replace', index=False)
            logger.info(f"Saved {len(transactions_df)} transactions to database")
        except Exception as e:
            logger.error(f"Error saving transactions: {e}")
            raise
    
    def save_customers_to_db(self, customers_df):
        """Save customer data to database"""
        try:
            customers_df.to_sql('customers', self.engine, if_exists='replace', index=False)
            logger.info(f"Saved {len(customers_df)} customers to database")
        except Exception as e:
            logger.error(f"Error saving customers: {e}")
            raise
    
    def load_transactions_from_db(self):
        """Load transaction data from database"""
        try:
            df = pd.read_sql('SELECT * FROM transactions', self.engine)
            logger.info(f"Loaded {len(df)} transactions from database")
            return df
        except Exception as e:
            logger.info("No transactions found in database, will use generated data")
            return None
    
    def load_customers_from_db(self):
        """Load customer data from database"""
        try:
            df = pd.read_sql('SELECT * FROM customers', self.engine)
            logger.info(f"Loaded {len(df)} customers from database")
            return df
        except Exception as e:
            logger.info("No customers found in database, will use generated data")
            return None
    
    def save_alert(self, alert_name, alert_type, threshold_value, operator='>', is_active=True):
        """Save user alert configuration"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO user_alerts (alert_name, alert_type, threshold_value, operator, is_active)
                    VALUES (:alert_name, :alert_type, :threshold_value, :operator, :is_active)
                """), {
                    'alert_name': alert_name,
                    'alert_type': alert_type,
                    'threshold_value': threshold_value,
                    'operator': operator,
                    'is_active': is_active
                })
                conn.commit()
                logger.info(f"Alert '{alert_name}' saved successfully")
        except Exception as e:
            logger.error(f"Error saving alert: {e}")
            raise
    
    def get_active_alerts(self):
        """Get all active alerts"""
        try:
            df = pd.read_sql("""
                SELECT * FROM user_alerts 
                WHERE is_active = TRUE 
                ORDER BY created_at DESC
            """, self.engine)
            return df
        except Exception as e:
            logger.error(f"Error fetching alerts: {e}")
            return pd.DataFrame()
    
    def save_report_history(self, report_name, report_type, file_path, parameters=None, file_size=None):
        """Save report generation history"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO report_history (report_name, report_type, file_path, parameters, file_size)
                    VALUES (:report_name, :report_type, :file_path, :parameters, :file_size)
                """), {
                    'report_name': report_name,
                    'report_type': report_type,
                    'file_path': file_path,
                    'parameters': str(parameters) if parameters else None,
                    'file_size': file_size
                })
                conn.commit()
                logger.info(f"Report history saved: {report_name}")
        except Exception as e:
            logger.error(f"Error saving report history: {e}")
            raise
    
    def save_ml_model_metadata(self, model_name, model_type, version, metrics, training_data_size, parameters):
        """Save ML model training metadata"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO ml_models (model_name, model_type, version, accuracy_score, 
                                         precision_score, recall_score, f1_score, training_data_size, 
                                         model_parameters, is_active)
                    VALUES (:model_name, :model_type, :version, :accuracy, :precision, 
                           :recall, :f1, :training_size, :parameters, TRUE)
                """), {
                    'model_name': model_name,
                    'model_type': model_type,
                    'version': version,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'training_size': training_data_size,
                    'parameters': str(parameters)
                })
                conn.commit()
                logger.info(f"ML model metadata saved: {model_name}")
        except Exception as e:
            logger.error(f"Error saving ML model metadata: {e}")
            raise
    
    def log_streaming_data(self, data_source, records_processed, processing_time_ms, status='success', error_message=None):
        """Log streaming data processing"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO streaming_data_log (data_source, records_processed, processing_time_ms, status, error_message)
                    VALUES (:data_source, :records_processed, :processing_time, :status, :error_message)
                """), {
                    'data_source': data_source,
                    'records_processed': records_processed,
                    'processing_time': processing_time_ms,
                    'status': status,
                    'error_message': error_message
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging streaming data: {e}")
    
    def save_export_job(self, job_name, export_type, export_format, file_path=None, 
                       scheduled_time=None, status='pending', record_count=None, file_size=None):
        """Save export job information"""
        if not self.engine:
            return
        
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO export_jobs (job_name, export_type, export_format, file_path, 
                                           scheduled_time, status, record_count, file_size, executed_at)
                    VALUES (:job_name, :export_type, :export_format, :file_path, 
                           :scheduled_time, :status, :record_count, :file_size, 
                           CASE WHEN :status = 'completed' THEN CURRENT_TIMESTAMP ELSE NULL END)
                """), {
                    'job_name': job_name,
                    'export_type': export_type,
                    'export_format': export_format,
                    'file_path': file_path,
                    'scheduled_time': scheduled_time,
                    'status': status,
                    'record_count': record_count,
                    'file_size': file_size
                })
                conn.commit()
                logger.info(f"Export job saved: {job_name}")
        except Exception as e:
            logger.error(f"Error saving export job: {e}")
            raise
    
    def get_session(self):
        """Get database session"""
        if self.SessionLocal:
            return self.SessionLocal()
        return None
    
    def close(self):
        """Close database connection"""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")

# Global database manager instance
db_manager = DatabaseManager()
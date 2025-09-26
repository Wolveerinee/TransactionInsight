import pandas as pd
import numpy as np
import time
import threading
from datetime import datetime, timedelta
import streamlit as st
from typing import Dict, List, Optional, Callable
import random
import json
from database import db_manager
from data_generator import BankingDataGenerator
import logging
import copy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeDataStreamer:
    """
    Real-time data streaming simulation for banking transactions
    Generates continuous data streams and updates dashboard in real-time
    """
    
    def __init__(self):
        self.is_streaming = False
        self.streaming_thread = None
        self.data_generator = BankingDataGenerator()
        self.stream_interval = 5  # seconds between updates
        self.batch_size = 10  # transactions per batch
        self.subscribers = []  # callback functions for data updates
        self._lock = threading.Lock()  # Thread safety lock
        self._stop_event = threading.Event()  # For clean shutdown
        
        # Initialize streaming metrics
        self.stream_metrics = {
            'total_records_processed': 0,
            'total_batches': 0,
            'average_processing_time': 0,
            'last_update': None,
            'error_count': 0,
            'uptime_start': None
        }
    
    def add_subscriber(self, callback: Callable):
        """Add a callback function to be notified of new data"""
        with self._lock:
            self.subscribers.append(callback)
    
    def remove_subscriber(self, callback: Callable):
        """Remove a callback function"""
        with self._lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def notify_subscribers(self, new_data: Dict):
        """Notify all subscribers of new data"""
        with self._lock:
            subscribers_copy = self.subscribers.copy()
        
        for callback in subscribers_copy:
            try:
                callback(new_data)
            except Exception as e:
                logger.error(f"Error notifying subscriber: {e}")
    
    def generate_streaming_batch(self) -> Dict:
        """Generate a batch of new transaction data"""
        try:
            start_time = time.time()
            
            # Generate new transactions
            customers = self.data_generator.generate_customers(num_customers=5)
            transactions = self.data_generator.generate_transactions(
                customers_df=customers,
                num_transactions=self.batch_size,
                start_date=datetime.now() - timedelta(hours=1),
                end_date=datetime.now()
            )
            
            # Add some realistic patterns for streaming data
            self._add_streaming_patterns(transactions)
            
            processing_time = (time.time() - start_time) * 1000  # ms
            
            # Update metrics with thread safety
            with self._lock:
                self.stream_metrics['total_records_processed'] += len(transactions)
                self.stream_metrics['total_batches'] += 1
                self.stream_metrics['average_processing_time'] = (
                    (self.stream_metrics['average_processing_time'] * (self.stream_metrics['total_batches'] - 1) + 
                     processing_time) / self.stream_metrics['total_batches']
                )
                self.stream_metrics['last_update'] = datetime.now()
            
            # Log to database with error handling
            try:
                db_manager.log_streaming_data(
                    data_source='real_time_stream',
                    records_processed=len(transactions),
                    processing_time_ms=int(processing_time),
                    status='success'
                )
            except Exception as db_error:
                logger.warning(f"Failed to log streaming data to database: {db_error}")
            
            return {
                'transactions': transactions,
                'customers': customers,
                'processing_time_ms': processing_time,
                'batch_number': self.stream_metrics['total_batches'],
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            with self._lock:
                self.stream_metrics['error_count'] += 1
            logger.error(f"Error generating streaming batch: {e}")
            
            # Log error to database with safety
            try:
                db_manager.log_streaming_data(
                    data_source='real_time_stream',
                    records_processed=0,
                    processing_time_ms=0,
                    status='error',
                    error_message=str(e)
                )
            except Exception as db_error:
                logger.warning(f"Failed to log error to database: {db_error}")
            
            return {}
    
    def _add_streaming_patterns(self, transactions_df: pd.DataFrame):
        """Add realistic patterns to streaming data"""
        current_hour = datetime.now().hour
        
        # Higher transaction amounts during business hours
        if 9 <= current_hour <= 17:
            business_hours_mask = np.random.random(len(transactions_df)) < 0.3
            transactions_df.loc[business_hours_mask, 'amount'] *= np.random.uniform(1.2, 2.0, business_hours_mask.sum())
        
        # Weekend shopping patterns
        if datetime.now().weekday() >= 5:  # Saturday or Sunday
            weekend_mask = np.random.random(len(transactions_df)) < 0.4
            weekend_categories = ['retail', 'entertainment', 'restaurant']
            transactions_df.loc[weekend_mask, 'category'] = np.random.choice(
                weekend_categories, size=weekend_mask.sum()
            )
        
        # Late night transactions (potential fraud indicators)
        if current_hour >= 23 or current_hour <= 3:
            late_night_mask = np.random.random(len(transactions_df)) < 0.1
            transactions_df.loc[late_night_mask, 'amount'] *= np.random.uniform(2.0, 5.0, late_night_mask.sum())
            transactions_df.loc[late_night_mask, 'is_fraud'] = True
    
    def start_streaming(self):
        """Start the real-time data streaming"""
        with self._lock:
            if self.is_streaming:
                logger.info("Streaming is already active")
                return
            
            self.is_streaming = True
            self.stream_metrics['uptime_start'] = datetime.now()
        
        self._stop_event.clear()
        self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
        self.streaming_thread.start()
        logger.info("Real-time data streaming started")
    
    def stop_streaming(self):
        """Stop the real-time data streaming"""
        with self._lock:
            self.is_streaming = False
        
        self._stop_event.set()  # Signal the thread to stop immediately
        
        if self.streaming_thread and self.streaming_thread.is_alive():
            self.streaming_thread.join(timeout=10)
        logger.info("Real-time data streaming stopped")
    
    def _streaming_loop(self):
        """Main streaming loop that runs in background thread"""
        while True:
            with self._lock:
                should_continue = self.is_streaming
            
            if not should_continue:
                break
                
            try:
                # Generate new batch of data
                batch_data = self.generate_streaming_batch()
                
                if batch_data:
                    # Notify all subscribers
                    self.notify_subscribers(batch_data)
                    
                    logger.info(f"Processed batch {batch_data['batch_number']} with {len(batch_data['transactions'])} transactions")
                
                # Wait for next interval or stop signal
                if self._stop_event.wait(timeout=self.stream_interval):
                    break  # Stop event was set
                
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                with self._lock:
                    self.stream_metrics['error_count'] += 1
                
                # Wait before retrying or stop if signaled
                if self._stop_event.wait(timeout=self.stream_interval):
                    break
    
    def get_stream_status(self) -> Dict:
        """Get current streaming status and metrics"""
        with self._lock:
            uptime = None
            if self.stream_metrics['uptime_start']:
                uptime = datetime.now() - self.stream_metrics['uptime_start']
            
            return {
                'is_active': self.is_streaming,
                'total_records_processed': self.stream_metrics['total_records_processed'],
                'total_batches': self.stream_metrics['total_batches'],
                'average_processing_time_ms': round(self.stream_metrics['average_processing_time'], 2),
                'last_update': self.stream_metrics['last_update'],
                'error_count': self.stream_metrics['error_count'],
                'uptime': str(uptime).split('.')[0] if uptime else None,
                'stream_interval_seconds': self.stream_interval,
                'batch_size': self.batch_size
            }
    
    def configure_stream(self, interval: Optional[int] = None, batch_size: Optional[int] = None):
        """Configure streaming parameters"""
        if interval:
            self.stream_interval = max(1, interval)  # Minimum 1 second
        if batch_size:
            self.batch_size = max(1, min(batch_size, 100))  # Between 1 and 100
        
        logger.info(f"Stream configured: interval={self.stream_interval}s, batch_size={self.batch_size}")

class StreamingDashboard:
    """
    Dashboard component for real-time streaming visualization
    """
    
    def __init__(self, streamer: RealTimeDataStreamer):
        self.streamer = streamer
        self.live_data = {
            'recent_transactions': pd.DataFrame(),
            'live_metrics': {},
            'fraud_alerts': [],
            'volume_spikes': []
        }
        self._data_lock = threading.Lock()  # Thread safety for live data
        self._subscriber_added = False
        
        # Subscribe to streaming updates
        self._ensure_subscriber()
    
    def _ensure_subscriber(self):
        """Ensure subscriber is added only once"""
        if not self._subscriber_added:
            self.streamer.add_subscriber(self._update_live_data)
            self._subscriber_added = True
    
    def _update_live_data(self, batch_data: Dict):
        """Update live dashboard data with new batch"""
        try:
            with self._data_lock:
                # Add new transactions to recent data
                new_transactions = batch_data['transactions']
                
                # Keep only last 1000 transactions for performance
                self.live_data['recent_transactions'] = pd.concat([
                    self.live_data['recent_transactions'], 
                    new_transactions
                ]).tail(1000).reset_index(drop=True)
                
                # Update live metrics
                self._calculate_live_metrics(new_transactions)
                
                # Check for fraud alerts
                self._check_fraud_alerts(new_transactions)
                
                # Check for volume spikes
                self._check_volume_spikes(new_transactions)
            
        except Exception as e:
            logger.error(f"Error updating live data: {e}")
    
    def get_live_data_snapshot(self) -> Dict:
        """Get a thread-safe snapshot of live data"""
        with self._data_lock:
            return {
                'recent_transactions': self.live_data['recent_transactions'].copy() if not self.live_data['recent_transactions'].empty else pd.DataFrame(),
                'live_metrics': copy.deepcopy(self.live_data['live_metrics']),
                'fraud_alerts': copy.deepcopy(self.live_data['fraud_alerts']),
                'volume_spikes': copy.deepcopy(self.live_data['volume_spikes'])
            }
    
    def _calculate_live_metrics(self, new_transactions: pd.DataFrame):
        """Calculate real-time metrics"""
        if len(new_transactions) == 0:
            return
        
        # Calculate current metrics
        current_metrics = {
            'transactions_per_minute': len(new_transactions) * (60 / self.streamer.stream_interval),
            'average_amount': new_transactions['amount'].mean(),
            'total_volume': new_transactions['amount'].sum(),
            'fraud_rate': (new_transactions['is_fraud'].sum() / len(new_transactions)) * 100,
            'unique_customers': new_transactions['customer_id'].nunique(),
            'top_category': new_transactions['category'].mode().iloc[0] if len(new_transactions) > 0 else 'Unknown',
            'timestamp': datetime.now()
        }
        
        self.live_data['live_metrics'] = current_metrics
    
    def _check_fraud_alerts(self, new_transactions: pd.DataFrame):
        """Check for fraud patterns in new transactions"""
        fraud_transactions = new_transactions[new_transactions['is_fraud'] == True]
        
        for _, transaction in fraud_transactions.iterrows():
            alert = {
                'timestamp': datetime.now(),
                'transaction_id': transaction['transaction_id'],
                'customer_id': transaction['customer_id'],
                'amount': transaction['amount'],
                'category': transaction['category'],
                'alert_type': 'fraud_detected',
                'severity': 'high' if transaction['amount'] > 1000 else 'medium'
            }
            
            self.live_data['fraud_alerts'].append(alert)
            
            # Keep only last 50 alerts
            self.live_data['fraud_alerts'] = self.live_data['fraud_alerts'][-50:]
    
    def _check_volume_spikes(self, new_transactions: pd.DataFrame):
        """Check for unusual transaction volume spikes"""
        if len(self.live_data['recent_transactions']) < 100:
            return
        
        # Get stream status with proper locking
        status = self.streamer.get_stream_status()
        
        # Calculate rolling averages
        recent_batch_size = len(new_transactions)
        historical_avg = len(self.live_data['recent_transactions']) / max(1, status['total_batches'])
        
        # Alert if current batch is significantly larger
        if recent_batch_size > historical_avg * 2:
            spike_alert = {
                'timestamp': datetime.now(),
                'current_volume': recent_batch_size,
                'average_volume': round(historical_avg, 2),
                'spike_ratio': round(recent_batch_size / historical_avg, 2),
                'alert_type': 'volume_spike'
            }
            
            self.live_data['volume_spikes'].append(spike_alert)
            
            # Keep only last 20 spikes
            self.live_data['volume_spikes'] = self.live_data['volume_spikes'][-20:]
    
    def render_live_dashboard(self):
        """Render the live streaming dashboard"""
        # Initialize session state for streaming control
        if 'streaming_active' not in st.session_state:
            st.session_state.streaming_active = False
        
        # Ensure subscriber is added
        self._ensure_subscriber()
        
        st.subheader("🔴 Live Data Stream")
        
        # Auto-refresh for continuous updates using Streamlit's auto_refresh
        if st.session_state.streaming_active:
            # Auto-refresh every 2 seconds when streaming is active
            st.empty()
            if hasattr(st, 'autorefresh'):
                st.autorefresh(interval=2000, key="streaming_refresh")
            else:
                # Fallback for older Streamlit versions
                import time
                refresh_placeholder = st.empty()
                with refresh_placeholder.container():
                    if 'last_refresh_time' not in st.session_state:
                        st.session_state.last_refresh_time = time.time()
                    
                    current_time = time.time()
                    if current_time - st.session_state.last_refresh_time > 2:  # Refresh every 2 seconds
                        st.session_state.last_refresh_time = current_time
                        st.rerun()
        
        # Stream status
        status = self.streamer.get_stream_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if status['is_active']:
                st.success("🟢 Stream Active")
            else:
                st.error("🔴 Stream Inactive")
        
        with col2:
            st.metric("Total Records", status['total_records_processed'])
        
        with col3:
            st.metric("Avg Processing Time", f"{status['average_processing_time_ms']:.1f}ms")
        
        with col4:
            st.metric("Uptime", status['uptime'] or "Not started")
        
        # Stream controls
        st.subheader("Stream Controls")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("▶️ Start Stream"):
                self.streamer.start_streaming()
                st.session_state.streaming_active = True
                st.rerun()
        
        with col2:
            if st.button("⏹️ Stop Stream"):
                self.streamer.stop_streaming()
                st.session_state.streaming_active = False
                st.rerun()
        
        with col3:
            if st.button("🔄 Refresh"):
                st.rerun()
        
        # Configuration
        with st.expander("Stream Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_interval = st.slider("Update Interval (seconds)", 1, 30, status['stream_interval_seconds'])
            
            with col2:
                new_batch_size = st.slider("Batch Size", 1, 50, status['batch_size'])
            
            if st.button("Apply Configuration"):
                self.streamer.configure_stream(new_interval, new_batch_size)
                st.success("Configuration updated!")
                st.rerun()
        
        # Get thread-safe data snapshot
        live_data_snapshot = self.get_live_data_snapshot()
        
        # Live metrics
        if live_data_snapshot['live_metrics']:
            st.subheader("Real-Time Metrics")
            metrics = live_data_snapshot['live_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Transactions/min", f"{metrics['transactions_per_minute']:.1f}")
            
            with col2:
                st.metric("Avg Amount", f"${metrics['average_amount']:.2f}")
            
            with col3:
                st.metric("Fraud Rate", f"{metrics['fraud_rate']:.1f}%")
            
            with col4:
                st.metric("Unique Customers", metrics['unique_customers'])
        
        # Recent transactions
        if len(live_data_snapshot['recent_transactions']) > 0:
            st.subheader("Recent Transactions")
            
            # Show latest 10 transactions
            recent_display = live_data_snapshot['recent_transactions'].tail(10)[
                ['transaction_id', 'customer_id', 'amount', 'category', 'timestamp', 'is_fraud']
            ].sort_values('timestamp', ascending=False)
            
            # Color code fraud transactions
            def highlight_fraud(row):
                return ['background-color: #ffcccc' if row['is_fraud'] else '' for _ in row]
            
            st.dataframe(
                recent_display.style.apply(highlight_fraud, axis=1),
                use_container_width=True
            )
        
        # Alerts
        if live_data_snapshot['fraud_alerts']:
            st.subheader("🚨 Fraud Alerts")
            
            for alert in live_data_snapshot['fraud_alerts'][-5:]:  # Show last 5 alerts
                severity_color = "🔴" if alert['severity'] == 'high' else "🟡"
                st.warning(
                    f"{severity_color} **Fraud Detected** - "
                    f"Transaction {alert['transaction_id']} "
                    f"(${alert['amount']:.2f}) at {alert['timestamp'].strftime('%H:%M:%S')}"
                )

# Global streaming instance
streaming_manager = RealTimeDataStreamer()
streaming_dashboard = StreamingDashboard(streaming_manager)
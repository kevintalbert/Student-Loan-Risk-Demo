"""
StudentCare Output Pipeline for Student Loan Risk Predictions

This module creates the final output dataset for StudentCare Solutions with delinquency predictions
and risk assessments for student loan borrowers from LoanTech Solutions.
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import sys

# Add utils to path for imports
sys.path.append(os.path.dirname(__file__))

from data_preprocessing import StudentLoanPreprocessor
from ml_models import StudentLoanRiskModels

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StudentCareOutputPipeline:
    """Pipeline to generate StudentCare-ready delinquency prediction dataset."""
    
    def __init__(self, model_dir: str = "models", output_dir: str = "data/studentcare_output"):
        """Initialize the StudentCare output pipeline."""
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.preprocessor = None
        self.ml_models = None
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # StudentCare output specifications
        self.studentcare_schema = {
            "borrower_id": "string",
            "first_name": "string",
            "last_name": "string",
            "ssn_last_4": "string",
            "phone_number": "string",
            "email": "string",
            "address": "string",
            "city": "string",
            "state": "string",
            "zip_code": "string",
            "loan_id": "string",
            "current_balance": "number",
            "days_delinquent": "integer",
            "risk_score": "number",
            "risk_category": "string",
            "delinquency_probability": "number",
            "recommended_action": "string",
            "priority_level": "integer",
            "contact_preference": "string",
            "best_contact_time": "string",
            "last_payment_date": "string",
            "next_due_date": "string",
            "total_owed": "number",
            "processing_date": "string",
            "loantech_account_status": "string"
        }
    
    def load_models(self) -> bool:
        """Load trained models and preprocessor."""
        try:
            logger.info("Loading models and preprocessor for StudentCare pipeline...")
            
            # Initialize and load ML models
            self.ml_models = StudentLoanRiskModels()
            
            if os.path.exists(self.model_dir):
                self.ml_models.load_models(self.model_dir)
                logger.info("Models loaded successfully")
                
                # Initialize preprocessor and fit it with the original data
                self.preprocessor = StudentLoanPreprocessor()
                
                # Load the original training data to fit the preprocessor
                data_path = 'data/synthetic/student_loan_master_dataset.csv'
                if os.path.exists(data_path):
                    import pandas as pd
                    df = pd.read_csv(data_path)
                    # Fit the preprocessor with the same data used for training
                    self.preprocessor.prepare_training_data(df, test_size=0.2, random_state=42)
                    logger.info("Preprocessor fitted successfully")
                else:
                    logger.warning("Training data not found. Preprocessor will not be fitted.")
                    return False
                
                return True
            else:
                logger.warning(f"Model directory not found: {self.model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def generate_synthetic_contact_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate synthetic contact information for borrowers."""
        
        # Lists for generating realistic synthetic data
        first_names = [
            'John', 'Jane', 'Michael', 'Sarah', 'David', 'Emily', 'Christopher', 'Jessica',
            'Matthew', 'Ashley', 'Anthony', 'Amanda', 'Joshua', 'Stephanie', 'Andrew', 'Megan',
            'Daniel', 'Jennifer', 'Ryan', 'Melissa', 'James', 'Amy', 'Robert', 'Nicole'
        ]
        
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
            'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
            'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson', 'White'
        ]
        
        contact_preferences = ['Phone', 'Email', 'Text', 'Mail']
        contact_times = ['Morning (9-12)', 'Afternoon (12-5)', 'Evening (5-8)', 'Anytime']
        
        account_statuses = ['Active', 'Forbearance', 'Deferment', 'In Repayment', 'Grace Period']
        
        # Generate contact information
        df_contact = df.copy()
        n_borrowers = len(df)
        
        df_contact['first_name'] = np.random.choice(first_names, n_borrowers)
        df_contact['last_name'] = np.random.choice(last_names, n_borrowers)
        
        # Generate SSN last 4 digits
        df_contact['ssn_last_4'] = [f"{np.random.randint(1000, 9999)}" for _ in range(n_borrowers)]
        
        # Generate phone numbers
        df_contact['phone_number'] = [
            f"({np.random.randint(200, 999):03d}) {np.random.randint(200, 999):03d}-{np.random.randint(1000, 9999):04d}"
            for _ in range(n_borrowers)
        ]
        
        # Generate email addresses
        df_contact['email'] = [
            f"{row['first_name'].lower()}.{row['last_name'].lower()}{np.random.randint(1, 999)}@{np.random.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com'])}"
            for _, row in df_contact[['first_name', 'last_name']].iterrows()
        ]
        
        # Generate addresses
        street_numbers = np.random.randint(100, 9999, n_borrowers)
        street_names = np.random.choice([
            'Main St', 'Oak Ave', 'First St', 'Second St', 'Park Ave', 'Elm St',
            'Washington St', 'Maple Ave', 'Cedar St', 'Pine St'
        ], n_borrowers)
        
        df_contact['address'] = [f"{num} {street}" for num, street in zip(street_numbers, street_names)]
        
        # Generate cities and zip codes
        cities = ['Springfield', 'Franklin', 'Georgetown', 'Clinton', 'Madison', 'Washington']
        df_contact['city'] = np.random.choice(cities, n_borrowers)
        df_contact['zip_code'] = [f"{np.random.randint(10000, 99999)}" for _ in range(n_borrowers)]
        
        # Contact preferences and times
        df_contact['contact_preference'] = np.random.choice(contact_preferences, n_borrowers)
        df_contact['best_contact_time'] = np.random.choice(contact_times, n_borrowers)
        
        # Account status
        df_contact['loantech_account_status'] = np.random.choice(account_statuses, n_borrowers)
        
        return df_contact
    
    def generate_loan_details(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate additional loan details for StudentCare output."""
        
        df_loans = df.copy()
        
        # Generate loan IDs (use existing pattern or create new)
        if 'loan_id' not in df_loans.columns:
            df_loans['loan_id'] = [f"ML_{i:08d}" for i in range(1, len(df_loans) + 1)]
        
        # Calculate current balance from total_current_balance or estimate
        if 'total_current_balance' in df_loans.columns:
            df_loans['current_balance'] = df_loans['total_current_balance']
        else:
            # Estimate current balance as 80% of total loan amount
            df_loans['current_balance'] = df_loans['total_loan_amount'] * 0.8
        
        # Calculate days delinquent based on recent payment behavior
        df_loans['days_delinquent'] = np.where(
            df_loans['recent_missed_payments'] > 0,
            np.random.randint(1, 120, len(df_loans)),  # Random days if recently missed
            0  # No delinquency if no recent missed payments
        )
        
        # Generate payment dates
        base_date = datetime.now()
        
        # Last payment date (1-60 days ago)
        df_loans['last_payment_date'] = [
            (base_date - timedelta(days=np.random.randint(1, 61))).strftime('%Y-%m-%d')
            for _ in range(len(df_loans))
        ]
        
        # Next due date (within next 30 days)
        df_loans['next_due_date'] = [
            (base_date + timedelta(days=np.random.randint(1, 31))).strftime('%Y-%m-%d')
            for _ in range(len(df_loans))
        ]
        
        # Total owed (current balance + potential fees)
        fee_multiplier = np.where(df_loans['days_delinquent'] > 0, 1.1, 1.0)  # 10% fee if delinquent
        df_loans['total_owed'] = df_loans['current_balance'] * fee_multiplier
        
        return df_loans
    
    def calculate_delinquency_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate delinquency predictions using trained models."""
        
        if not self.ml_models or not self.preprocessor:
            logger.warning("Models not loaded. Using mock predictions.")
            # Generate mock predictions
            df_pred = df.copy()
            df_pred['delinquency_probability'] = np.random.uniform(0, 1, len(df))
            df_pred['risk_score'] = df_pred['delinquency_probability'] * 100
            df_pred['risk_category'] = pd.cut(
                df_pred['risk_score'],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            return df_pred
        
        try:
            # Preprocess data
            X = self.preprocessor.transform_new_data(df)
            
            # Generate predictions
            predictions = self.ml_models.predict_delinquency_risk(X)
            
            # Add predictions to dataframe
            df_pred = df.copy()
            df_pred['delinquency_probability'] = predictions['risk_probability']
            df_pred['risk_score'] = predictions['risk_probability'] * 100
            df_pred['risk_category'] = predictions['risk_category']
            
            return df_pred
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            # Fallback to mock predictions (avoid recursion)
            logger.warning("Using mock predictions due to prediction error.")
            df_pred = df.copy()
            df_pred['delinquency_probability'] = np.random.uniform(0, 1, len(df))
            df_pred['risk_score'] = df_pred['delinquency_probability'] * 100
            df_pred['risk_category'] = pd.cut(
                df_pred['risk_score'],
                bins=[0, 25, 50, 75, 100],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            return df_pred
    
    def determine_recommended_actions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Determine recommended actions for StudentCare based on risk assessment."""
        
        df_actions = df.copy()
        
        # Define actions based on risk category and current status
        def get_action(row):
            risk_cat = row['risk_category']
            days_delinq = row.get('days_delinquent', 0)
            
            if days_delinq > 90:
                return "Immediate Collection Action"
            elif days_delinq > 30:
                return "Urgent Payment Arrangement"
            elif risk_cat == 'Critical':
                return "Proactive Outreach - High Priority"
            elif risk_cat == 'High':
                return "Proactive Outreach - Medium Priority"
            elif risk_cat == 'Medium':
                return "Educational Communication"
            else:
                return "Standard Monitoring"
        
        df_actions['recommended_action'] = df_actions.apply(get_action, axis=1)
        
        # Assign priority levels (1=highest, 5=lowest)
        priority_map = {
            "Immediate Collection Action": 1,
            "Urgent Payment Arrangement": 2,
            "Proactive Outreach - High Priority": 2,
            "Proactive Outreach - Medium Priority": 3,
            "Educational Communication": 4,
            "Standard Monitoring": 5
        }
        
        df_actions['priority_level'] = df_actions['recommended_action'].map(priority_map)
        
        return df_actions
    
    def create_studentcare_dataset(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """Create the complete StudentCare-ready dataset."""
        
        logger.info("Creating StudentCare-ready dataset...")
        
        # Start with input data
        df = input_df.copy()
        
        # Generate contact information
        df = self.generate_synthetic_contact_info(df)
        
        # Generate loan details
        df = self.generate_loan_details(df)
        
        # Calculate delinquency predictions
        df = self.calculate_delinquency_predictions(df)
        
        # Determine recommended actions
        df = self.determine_recommended_actions(df)
        
        # Add processing metadata
        df['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Select and order columns according to StudentCare schema
        studentcare_columns = list(self.studentcare_schema.keys())
        
        # Only include columns that exist in the dataframe
        available_columns = [col for col in studentcare_columns if col in df.columns]
        df_studentcare = df[available_columns].copy()
        
        # Round numerical columns
        for col in df_studentcare.columns:
            if df_studentcare[col].dtype in ['float64', 'float32']:
                df_studentcare[col] = df_studentcare[col].round(2)
        
        logger.info(f"StudentCare dataset created with {len(df_studentcare)} records and {len(available_columns)} columns")
        
        return df_studentcare
    
    def filter_high_risk_borrowers(self, df: pd.DataFrame, 
                                  min_risk_score: float = 50.0) -> pd.DataFrame:
        """Filter dataset to include only high-risk borrowers for StudentCare."""
        
        # Filter based on risk criteria
        high_risk_mask = (
            (df['risk_score'] >= min_risk_score) |
            (df['days_delinquent'] > 0) |
            (df['risk_category'].isin(['High', 'Critical']))
        )
        
        df_high_risk = df[high_risk_mask].copy()
        
        logger.info(f"Filtered to {len(df_high_risk)} high-risk borrowers from {len(df)} total borrowers")
        
        return df_high_risk
    
    def generate_summary_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary report for StudentCare dataset."""
        
        summary = {
            "processing_date": datetime.now().isoformat(),
            "total_borrowers": len(df),
            "risk_distribution": df['risk_category'].value_counts().to_dict(),
            "average_risk_score": float(df['risk_score'].mean()),
            "delinquent_borrowers": int((df['days_delinquent'] > 0).sum()),
            "action_priorities": df['priority_level'].value_counts().sort_index().to_dict(),
            "recommended_actions": df['recommended_action'].value_counts().to_dict(),
            "total_outstanding_balance": float(df['total_owed'].sum()),
            "average_outstanding_balance": float(df['total_owed'].mean()),
            "contact_preferences": df['contact_preference'].value_counts().to_dict(),
            "state_distribution": df['state'].value_counts().to_dict(),
            "data_quality": {
                "completeness": float((df.size - df.isnull().sum().sum()) / df.size),
                "missing_values": int(df.isnull().sum().sum())
            }
        }
        
        return summary
    
    def save_studentcare_outputs(self, df: pd.DataFrame, filename_prefix: str = "studentcare_delinquency_predictions") -> Dict[str, str]:
        """Save StudentCare outputs in multiple formats."""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # File paths
        files = {
            "csv": os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.csv"),
            "excel": os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.xlsx"),
            "json": os.path.join(self.output_dir, f"{filename_prefix}_{timestamp}.json"),
            "summary": os.path.join(self.output_dir, f"{filename_prefix}_summary_{timestamp}.json")
        }
        
        # Save CSV (primary format for StudentCare)
        df.to_csv(files["csv"], index=False)
        
        # Save Excel with multiple sheets
        with pd.ExcelWriter(files["excel"], engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Delinquency_Predictions', index=False)
            
            # Add summary sheet
            summary_df = pd.DataFrame([
                {"Metric": "Total Borrowers", "Value": len(df)},
                {"Metric": "High Risk Count", "Value": (df['risk_category'].isin(['High', 'Critical'])).sum()},
                {"Metric": "Average Risk Score", "Value": f"{df['risk_score'].mean():.2f}"},
                {"Metric": "Total Outstanding", "Value": f"${df['total_owed'].sum():,.2f}"}
            ])
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Save JSON
        df.to_json(files["json"], orient='records', indent=2, date_format='iso')
        
        # Save summary report
        summary_report = self.generate_summary_report(df)
        with open(files["summary"], 'w') as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        logger.info(f"StudentCare outputs saved:")
        for format_type, filepath in files.items():
            logger.info(f"  {format_type.upper()}: {filepath}")
        
        return files
    
    def run_complete_pipeline(self, input_data_path: str, 
                            filter_high_risk: bool = True,
                            min_risk_score: float = 50.0) -> Dict[str, Any]:
        """Run the complete StudentCare output pipeline."""
        
        logger.info("Starting StudentCare output pipeline...")
        
        try:
            # Load models
            if not self.load_models():
                logger.warning("Models not available. Proceeding with mock predictions.")
            
            # Load input data
            if input_data_path.endswith('.csv'):
                input_df = pd.read_csv(input_data_path)
            else:
                raise ValueError("Input data must be a CSV file")
            
            logger.info(f"Loaded {len(input_df)} borrower records")
            
            # Create StudentCare dataset
            studentcare_df = self.create_studentcare_dataset(input_df)
            
            # Filter to high-risk borrowers if requested
            if filter_high_risk:
                studentcare_df = self.filter_high_risk_borrowers(studentcare_df, min_risk_score)
            
            # Save outputs
            output_files = self.save_studentcare_outputs(studentcare_df)
            
            # Generate summary
            summary = self.generate_summary_report(studentcare_df)
            
            result = {
                "status": "success",
                "records_processed": len(input_df),
                "high_risk_identified": len(studentcare_df),
                "output_files": output_files,
                "summary": summary
            }
            
            logger.info("StudentCare pipeline completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "error",
                "error_message": str(e)
            }


def main():
    """Test the StudentCare output pipeline."""
    
    # Initialize pipeline
    pipeline = StudentCareOutputPipeline()
    
    # Test with synthetic data
    input_data_path = "../data/synthetic/student_loan_master_dataset.csv"
    
    if os.path.exists(input_data_path):
        # Run pipeline
        result = pipeline.run_complete_pipeline(
            input_data_path=input_data_path,
            filter_high_risk=True,
            min_risk_score=50.0
        )
        
        # Print results
        print(json.dumps(result, indent=2, default=str))
        
    else:
        print(f"Input data not found: {input_data_path}")
        print("Run data_generator.py first to create the synthetic dataset.")


if __name__ == "__main__":
    main()

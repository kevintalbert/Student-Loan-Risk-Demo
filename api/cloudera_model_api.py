"""
Cloudera Model API Integration for Student Loan Risk Prediction

This module implements the Cloudera Model API for deploying and serving
student loan delinquency risk prediction models in Cloudera Machine Learning.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import traceback

# Add utils to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Try to import CML-specific modules (optional for local development)
try:
    import cdsw
    CML_AVAILABLE = True
except ImportError:
    CML_AVAILABLE = False
    print("Note: CDSW module not available. Running in local development mode.")

try:
    from data_preprocessing import StudentLoanPreprocessor
    from ml_models import StudentLoanRiskModels
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    print("Make sure utils modules are in the Python path")


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClouderaStudentLoanRiskAPI:
    """Cloudera Model API wrapper for student loan risk prediction."""
    
    def __init__(self, model_dir: str = "../models"):
        """Initialize the API with trained models and preprocessor."""
        self.model_dir = model_dir
        self.preprocessor = None
        self.ml_models = None
        self.is_loaded = False
        
        # Model metadata
        self.model_info = {
            "name": "Student Loan Delinquency Risk Predictor",
            "version": "1.0.0",
            "description": "ML model to predict student loan delinquency risk for LoanTech Solutions/StudentCare Solutions",
            "created_date": datetime.now().isoformat(),
            "input_schema": self._get_input_schema(),
            "output_schema": self._get_output_schema()
        }
    
    def _get_input_schema(self) -> Dict[str, Any]:
        """Define the expected input schema for the API."""
        return {
            "type": "object",
            "properties": {
                "borrower_id": {"type": "string", "description": "Unique borrower identifier"},
                "age": {"type": "integer", "minimum": 18, "maximum": 80},
                "gender": {"type": "string", "enum": ["M", "F", "O"]},
                "state": {"type": "string", "description": "Two-letter state code"},
                "credit_score_at_origination": {"type": "integer", "minimum": 300, "maximum": 850},
                "annual_income": {"type": "number", "minimum": 0},
                "employment_status": {"type": "string", "enum": ["Employed Full-time", "Employed Part-time", "Unemployed", "Student"]},
                "dependents": {"type": "integer", "minimum": 0},
                "housing_status": {"type": "string", "enum": ["Own", "Rent", "Family"]},
                "school_name": {"type": "string"},
                "degree_type": {"type": "string", "enum": ["Associates", "Bachelors", "Masters", "Doctorate", "Certificate"]},
                "major": {"type": "string"},
                "graduation_year": {"type": "integer", "minimum": 1990, "maximum": 2030},
                "gpa": {"type": "number", "minimum": 0.0, "maximum": 4.0},
                "school_type": {"type": "string", "enum": ["Public", "Private"]},
                "completion_status": {"type": "string", "enum": ["Completed", "Dropped Out", "Transferred"]},
                "total_loan_amount": {"type": "number", "minimum": 0},
                "loan_count": {"type": "integer", "minimum": 1},
                "avg_interest_rate": {"type": "number", "minimum": 0},
                "total_current_balance": {"type": "number", "minimum": 0},
                "total_monthly_payment": {"type": "number", "minimum": 0},
                "avg_days_late": {"type": "number", "minimum": 0},
                "max_days_late": {"type": "number", "minimum": 0},
                "missed_payment_count": {"type": "integer", "minimum": 0},
                "recent_missed_payments": {"type": "integer", "minimum": 0}
            },
            "required": [
                "borrower_id", "age", "credit_score_at_origination", "annual_income",
                "total_loan_amount", "loan_count", "total_monthly_payment"
            ]
        }
    
    def _get_output_schema(self) -> Dict[str, Any]:
        """Define the output schema for the API."""
        return {
            "type": "object",
            "properties": {
                "borrower_id": {"type": "string"},
                "risk_probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "risk_prediction": {"type": "integer", "enum": [0, 1]},
                "risk_category": {"type": "string", "enum": ["Low", "Medium", "High", "Critical"]},
                "risk_score": {"type": "number", "minimum": 0, "maximum": 100},
                "prediction_timestamp": {"type": "string", "format": "date-time"},
                "model_version": {"type": "string"},
                "confidence_interval": {
                    "type": "object",
                    "properties": {
                        "lower_bound": {"type": "number"},
                        "upper_bound": {"type": "number"}
                    }
                }
            }
        }
    
    def load_models(self) -> bool:
        """Load trained models and preprocessor."""
        try:
            logger.info("Loading models and preprocessor...")
            
            # Initialize preprocessor
            self.preprocessor = StudentLoanPreprocessor()
            
            # Initialize ML models
            self.ml_models = StudentLoanRiskModels()
            
            # Load models from disk
            if os.path.exists(self.model_dir):
                self.ml_models.load_models(self.model_dir)
                logger.info("Models loaded successfully")
                self.is_loaded = True
                return True
            else:
                logger.warning(f"Model directory not found: {self.model_dir}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main prediction function for Cloudera Model API.
        
        Args:
            input_data: Dictionary containing borrower information
            
        Returns:
            Dictionary containing risk prediction results
        """
        try:
            # Ensure models are loaded
            if not self.is_loaded:
                if not self.load_models():
                    raise Exception("Failed to load models")
            
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                # Single prediction
                df = pd.DataFrame([input_data])
            elif isinstance(input_data, list):
                # Batch prediction
                df = pd.DataFrame(input_data)
            else:
                raise ValueError("Input must be a dictionary or list of dictionaries")
            
            # Validate input
            self._validate_input(df)
            
            # Preprocess data
            X = self.preprocessor.transform_new_data(df)
            
            # Generate predictions
            predictions = self.ml_models.predict_delinquency_risk(X)
            
            # Format output
            results = []
            for idx, row in df.iterrows():
                pred_row = predictions.iloc[idx]
                
                result = {
                    "borrower_id": row.get('borrower_id', f'UNKNOWN_{idx}'),
                    "risk_probability": float(pred_row['risk_probability']),
                    "risk_prediction": int(pred_row['risk_prediction']),
                    "risk_category": str(pred_row['risk_category']),
                    "risk_score": float(pred_row['risk_probability'] * 100),
                    "prediction_timestamp": datetime.now().isoformat(),
                    "model_version": self.model_info["version"],
                    "confidence_interval": self._calculate_confidence_interval(pred_row['risk_probability'])
                }
                results.append(result)
            
            # Return single result or list based on input
            if isinstance(input_data, dict):
                return results[0]
            else:
                return results
                
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            logger.error(traceback.format_exc())
            
            error_result = {
                "error": str(e),
                "prediction_timestamp": datetime.now().isoformat(),
                "model_version": self.model_info["version"]
            }
            
            return error_result
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input data against schema."""
        required_fields = [
            "borrower_id", "age", "credit_score_at_origination", "annual_income",
            "total_loan_amount", "loan_count", "total_monthly_payment"
        ]
        
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"Required field missing: {field}")
    
    def _calculate_confidence_interval(self, probability: float, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate confidence interval for prediction."""
        # Simplified confidence interval calculation
        # In practice, this would use more sophisticated methods
        margin = 0.1 * (1 - confidence)  # Simple margin calculation
        
        lower_bound = max(0.0, probability - margin)
        upper_bound = min(1.0, probability + margin)
        
        return {
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4)
        }
    
    def batch_predict(self, input_file: str, output_file: str) -> Dict[str, Any]:
        """Process batch predictions from file."""
        try:
            # Read input data
            if input_file.endswith('.csv'):
                df = pd.read_csv(input_file)
            elif input_file.endswith('.json'):
                df = pd.read_json(input_file)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            # Generate predictions
            input_data = df.to_dict('records')
            results = self.predict(input_data)
            
            # Save results
            results_df = pd.DataFrame(results)
            
            if output_file.endswith('.csv'):
                results_df.to_csv(output_file, index=False)
            elif output_file.endswith('.json'):
                results_df.to_json(output_file, orient='records', indent=2)
            
            summary = {
                "status": "success",
                "total_records": len(results),
                "high_risk_count": len([r for r in results if r.get('risk_category') in ['High', 'Critical']]),
                "average_risk_score": float(results_df['risk_score'].mean()),
                "output_file": output_file
            }
            
            logger.info(f"Batch prediction completed: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Batch prediction error: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information and metadata."""
        info = self.model_info.copy()
        
        if self.is_loaded and self.ml_models:
            info.update({
                "best_model": self.ml_models.best_model_name,
                "feature_count": len(self.ml_models.feature_names),
                "model_performance": self.ml_models.model_scores
            })
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Health check endpoint for monitoring."""
        try:
            # Check if models are loaded
            models_loaded = self.is_loaded
            
            # Simple prediction test
            test_data = {
                "borrower_id": "TEST_001",
                "age": 25,
                "credit_score_at_origination": 650,
                "annual_income": 50000,
                "total_loan_amount": 25000,
                "loan_count": 1,
                "total_monthly_payment": 250,
                "employment_status": "Employed Full-time",
                "degree_type": "Bachelors",
                "avg_days_late": 0,
                "max_days_late": 0,
                "missed_payment_count": 0,
                "recent_missed_payments": 0
            }
            
            if models_loaded:
                result = self.predict(test_data)
                prediction_test = "error" not in result
            else:
                prediction_test = False
            
            return {
                "status": "healthy" if models_loaded and prediction_test else "unhealthy",
                "models_loaded": models_loaded,
                "prediction_test": prediction_test,
                "timestamp": datetime.now().isoformat(),
                "version": self.model_info["version"]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Cloudera Model API entry points
def init():
    """Initialize the model (Cloudera Model API function)."""
    global api_instance
    api_instance = ClouderaStudentLoanRiskAPI()
    success = api_instance.load_models()
    
    if success:
        logger.info("Student Loan Risk API initialized successfully")
    else:
        logger.error("Failed to initialize Student Loan Risk API")
    
    return success


def predict(input_data):
    """Prediction function for Cloudera Model API."""
    global api_instance
    return api_instance.predict(input_data)


def health():
    """Health check function for Cloudera Model API."""
    global api_instance
    return api_instance.health_check()


def model_info():
    """Model information function for Cloudera Model API."""
    global api_instance
    return api_instance.get_model_info()


# Global API instance
api_instance = None


def main():
    """Test the API locally."""
    
    # Initialize API
    api = ClouderaStudentLoanRiskAPI()
    
    if not api.load_models():
        print("Warning: Could not load models. Testing with mock data...")
    
    # Test single prediction
    test_borrower = {
        "borrower_id": "BOR_000001",
        "age": 28,
        "gender": "F",
        "state": "CA",
        "credit_score_at_origination": 680,
        "annual_income": 65000,
        "employment_status": "Employed Full-time",
        "dependents": 1,
        "housing_status": "Rent",
        "school_name": "State University",
        "degree_type": "Bachelors",
        "major": "Business",
        "graduation_year": 2020,
        "gpa": 3.4,
        "school_type": "Public",
        "completion_status": "Completed",
        "total_loan_amount": 35000,
        "loan_count": 2,
        "avg_loan_amount": 17500,
        "avg_interest_rate": 5.2,
        "total_current_balance": 32000,
        "total_monthly_payment": 380,
        "debt_to_income_ratio": 0.070,
        "avg_days_late": 2.5,
        "max_days_late": 15,
        "std_days_late": 5.2,
        "total_payments_made": 8000,
        "total_payment_count": 24,
        "total_scheduled": 9120,
        "missed_payment_count": 1,
        "payment_ratio": 0.88,
        "missed_payment_rate": 0.042,
        "recent_avg_days_late": 0,
        "recent_missed_payments": 0,
        "risk_score": 25.5
    }
    
    print("Testing single prediction...")
    result = api.predict(test_borrower)
    print(json.dumps(result, indent=2))
    
    # Test health check
    print("\nTesting health check...")
    health_result = api.health_check()
    print(json.dumps(health_result, indent=2))
    
    # Test model info
    print("\nTesting model info...")
    info_result = api.get_model_info()
    print(json.dumps(info_result, indent=2))


if __name__ == "__main__":
    main()

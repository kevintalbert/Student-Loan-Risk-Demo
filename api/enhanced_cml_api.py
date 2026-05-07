"""
Enhanced Cloudera ML Model API - Best Practices Implementation

This version incorporates CML best practices and industry standards for
production model deployment.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import traceback

# CML-specific imports
try:
    import cdsw
    CML_AVAILABLE = True
except ImportError:
    CML_AVAILABLE = False
    print("Warning: CDSW module not available. Running in local mode.")

# Configure logging for CML
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CML Environment Configuration
MODEL_PATH = os.getenv('CDSW_MODEL_PATH', '/home/cdsw/models')
PROJECT_PATH = os.getenv('CDSW_PROJECT_PATH', '/home/cdsw')
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Add project utils to path
sys.path.append(os.path.join(PROJECT_PATH, 'utils'))

# Global variables for CML
model_instance = None
preprocessor_instance = None
model_metadata = {}

def init():
    """
    Initialize the model (Required CML function).
    
    This function is called once when the model starts up.
    It should load all necessary model artifacts and dependencies.
    
    Returns:
        bool: True if initialization successful, False otherwise
    """
    global model_instance, preprocessor_instance, model_metadata
    
    try:
        logger.info("Initializing Student Loan Risk Model...")
        
        # Log CML environment info
        if CML_AVAILABLE:
            logger.info(f"CML Model ID: {os.getenv('CDSW_MODEL_ID', 'N/A')}")
            logger.info(f"CML Model Version: {os.getenv('CDSW_MODEL_VERSION', 'N/A')}")
        
        # Import required modules
        from data_preprocessing import StudentLoanPreprocessor
        from ml_models import StudentLoanRiskModels
        
        # Initialize preprocessor
        preprocessor_instance = StudentLoanPreprocessor()
        logger.info("Preprocessor initialized")
        
        # Initialize and load ML models
        model_instance = StudentLoanRiskModels()
        
        if os.path.exists(MODEL_PATH):
            model_instance.load_models(MODEL_PATH)
            logger.info(f"Models loaded from {MODEL_PATH}")
        else:
            logger.warning(f"Model path not found: {MODEL_PATH}")
            return False
        
        # Load model metadata
        metadata_path = os.path.join(MODEL_PATH, 'model_metadata.joblib')
        if os.path.exists(metadata_path):
            model_metadata = joblib.load(metadata_path)
            logger.info("Model metadata loaded")
        
        # Performance optimization: Warm up the model
        _warm_up_model()
        
        logger.info("Model initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Model initialization failed: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def predict(args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction function (Required CML function).
    
    Args:
        args: Dictionary containing input data for prediction
        
    Returns:
        Dictionary containing prediction results
    """
    prediction_start_time = datetime.now()
    
    try:
        # Log prediction request (for monitoring)
        logger.info(f"Prediction request received: {prediction_start_time}")
        
        # Validate model is loaded
        if model_instance is None or preprocessor_instance is None:
            raise RuntimeError("Model not properly initialized")
        
        # Handle both single predictions and batch predictions
        if isinstance(args, dict):
            if 'data' in args:
                # Batch prediction format: {"data": [record1, record2, ...]}
                input_data = args['data']
                is_batch = True
            else:
                # Single prediction format: direct record
                input_data = args
                is_batch = False
        else:
            input_data = args
            is_batch = False
        
        # Convert to DataFrame
        if is_batch:
            df = pd.DataFrame(input_data)
        else:
            df = pd.DataFrame([input_data])
        
        # Validate required fields
        _validate_input(df)
        
        # Preprocess data
        X = preprocessor_instance.transform_new_data(df)
        
        # Generate predictions
        predictions = model_instance.predict_delinquency_risk(X)
        
        # Format results
        results = []
        for idx, row in df.iterrows():
            pred_row = predictions.iloc[idx]
            
            result = {
                "borrower_id": row.get('borrower_id', f'UNKNOWN_{idx}'),
                "risk_probability": float(pred_row['risk_probability']),
                "risk_prediction": int(pred_row['risk_prediction']),
                "risk_category": str(pred_row['risk_category']),
                "risk_score": float(pred_row['risk_probability'] * 100),
                "prediction_timestamp": prediction_start_time.isoformat(),
                "model_version": model_metadata.get('best_model_name', 'unknown'),
                "response_time_ms": int((datetime.now() - prediction_start_time).total_seconds() * 1000),
                "confidence_interval": _calculate_confidence_interval(pred_row['risk_probability'])
            }
            results.append(result)
        
        # Log successful prediction
        response_time = (datetime.now() - prediction_start_time).total_seconds()
        logger.info(f"Prediction completed in {response_time:.3f}s for {len(results)} records")
        
        # Return single result or batch results
        if is_batch or len(results) > 1:
            return {"predictions": results, "batch_size": len(results)}
        else:
            return results[0]
            
    except Exception as e:
        # Log error for monitoring
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Return error response
        error_response = {
            "error": str(e),
            "error_type": type(e).__name__,
            "prediction_timestamp": prediction_start_time.isoformat(),
            "model_version": model_metadata.get('best_model_name', 'unknown'),
            "response_time_ms": int((datetime.now() - prediction_start_time).total_seconds() * 1000)
        }
        
        return error_response


def health() -> Dict[str, Any]:
    """
    Health check function (CML standard function).
    
    Returns:
        Dictionary containing health status and diagnostics
    """
    try:
        health_check_time = datetime.now()
        
        # Check model status
        models_loaded = model_instance is not None and preprocessor_instance is not None
        
        # Perform a quick prediction test
        test_successful = False
        if models_loaded:
            try:
                test_data = {
                    "borrower_id": "HEALTH_CHECK",
                    "age": 25,
                    "credit_score_at_origination": 650,
                    "annual_income": 50000,
                    "total_loan_amount": 25000,
                    "loan_count": 1,
                    "total_monthly_payment": 250
                }
                
                result = predict(test_data)
                test_successful = "error" not in result
                
            except Exception as e:
                logger.warning(f"Health check prediction test failed: {str(e)}")
        
        # System health metrics
        health_status = {
            "status": "healthy" if models_loaded and test_successful else "unhealthy",
            "timestamp": health_check_time.isoformat(),
            "checks": {
                "models_loaded": models_loaded,
                "prediction_test": test_successful,
                "model_path_exists": os.path.exists(MODEL_PATH)
            },
            "model_info": {
                "best_model": model_metadata.get('best_model_name', 'unknown'),
                "feature_count": len(model_metadata.get('feature_names', [])),
                "version": model_metadata.get('version', '1.0.0')
            },
            "environment": {
                "cml_available": CML_AVAILABLE,
                "python_version": sys.version.split()[0],
                "log_level": LOG_LEVEL
            }
        }
        
        # Add CML-specific info if available
        if CML_AVAILABLE:
            health_status["cml_info"] = {
                "model_id": os.getenv('CDSW_MODEL_ID', 'N/A'),
                "model_version": os.getenv('CDSW_MODEL_VERSION', 'N/A'),
                "replica_id": os.getenv('CDSW_MODEL_REPLICA_ID', 'N/A')
            }
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def model_info() -> Dict[str, Any]:
    """
    Model information function (CML standard function).
    
    Returns:
        Dictionary containing detailed model information
    """
    try:
        info = {
            "name": "Student Loan Delinquency Risk Predictor",
            "version": "1.0.0",
            "description": "ML model to predict student loan delinquency risk for LoanTech Solutions/StudentCare Solutions",
            "framework": "scikit-learn",
            "created_date": datetime.now().isoformat(),
            "input_schema": _get_input_schema(),
            "output_schema": _get_output_schema()
        }
        
        # Add model performance metrics if available
        if model_metadata and 'model_scores' in model_metadata:
            info["performance"] = model_metadata['model_scores']
        
        # Add feature information
        if model_metadata and 'feature_names' in model_metadata:
            info["features"] = {
                "count": len(model_metadata['feature_names']),
                "names": model_metadata['feature_names'][:10]  # Top 10 for brevity
            }
        
        return info
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return {"error": str(e)}


# Helper functions
def _warm_up_model():
    """Warm up the model with a dummy prediction to improve first-call latency."""
    try:
        dummy_data = {
            "borrower_id": "WARMUP",
            "age": 25,
            "credit_score_at_origination": 650,
            "annual_income": 50000,
            "total_loan_amount": 25000,
            "loan_count": 1,
            "total_monthly_payment": 250
        }
        predict(dummy_data)
        logger.info("Model warm-up completed")
    except Exception as e:
        logger.warning(f"Model warm-up failed: {str(e)}")


def _validate_input(df: pd.DataFrame) -> None:
    """Validate input data against required schema."""
    required_fields = [
        "borrower_id", "age", "credit_score_at_origination", "annual_income",
        "total_loan_amount", "loan_count", "total_monthly_payment"
    ]
    
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"Required fields missing: {missing_fields}")


def _calculate_confidence_interval(probability: float, confidence: float = 0.95) -> Dict[str, float]:
    """Calculate confidence interval for prediction."""
    margin = 0.1 * (1 - confidence)
    return {
        "lower_bound": round(max(0.0, probability - margin), 4),
        "upper_bound": round(min(1.0, probability + margin), 4)
    }


def _get_input_schema() -> Dict[str, Any]:
    """Define the expected input schema."""
    return {
        "type": "object",
        "properties": {
            "borrower_id": {"type": "string", "description": "Unique borrower identifier"},
            "age": {"type": "integer", "minimum": 18, "maximum": 80},
            "credit_score_at_origination": {"type": "integer", "minimum": 300, "maximum": 850},
            "annual_income": {"type": "number", "minimum": 0},
            "total_loan_amount": {"type": "number", "minimum": 0},
            "loan_count": {"type": "integer", "minimum": 1},
            "total_monthly_payment": {"type": "number", "minimum": 0}
        },
        "required": [
            "borrower_id", "age", "credit_score_at_origination", "annual_income",
            "total_loan_amount", "loan_count", "total_monthly_payment"
        ]
    }


def _get_output_schema() -> Dict[str, Any]:
    """Define the output schema."""
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
            "response_time_ms": {"type": "integer"},
            "confidence_interval": {
                "type": "object",
                "properties": {
                    "lower_bound": {"type": "number"},
                    "upper_bound": {"type": "number"}
                }
            }
        }
    }


if __name__ == "__main__":
    # Local testing
    print("Testing Enhanced CML API locally...")
    
    if init():
        print("✓ Initialization successful")
        
        # Test prediction
        test_data = {
            "borrower_id": "TEST_001",
            "age": 28,
            "credit_score_at_origination": 680,
            "annual_income": 65000,
            "total_loan_amount": 35000,
            "loan_count": 2,
            "total_monthly_payment": 380
        }
        
        result = predict(test_data)
        print(f"✓ Prediction: {json.dumps(result, indent=2)}")
        
        # Test health check
        health_result = health()
        print(f"✓ Health: {health_result['status']}")
        
    else:
        print("✗ Initialization failed")

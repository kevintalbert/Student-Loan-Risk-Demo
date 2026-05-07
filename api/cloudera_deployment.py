"""
Cloudera Model Deployment Configuration

This module provides deployment utilities and configuration for deploying
the student loan risk model to Cloudera Machine Learning platform.
"""

import os
import json
import yaml
from typing import Dict, Any, List


class ClouderaModelDeployment:
    """Configuration and deployment utilities for Cloudera ML."""
    
    def __init__(self, project_name: str = "student-loan-risk"):
        """Initialize deployment configuration."""
        self.project_name = project_name
        
        # Cloudera ML configuration
        self.config = {
            "model": {
                "name": "student-loan-delinquency-predictor",
                "description": "ML model to predict student loan delinquency risk for LoanTech Solutions/StudentCare Solutions partnership",
                "version": "1.0.0",
                "framework": "scikit-learn",
                "python_version": "3.11",
                "entry_point": "cloudera_model_api.py"
            },
            "resources": {
                "cpu": 2,
                "memory": "4Gi",
                "gpu": 0
            },
            "environment": {
                "name": "student-loan-env",
                "conda_packages": [
                    "pandas>=2.1.0",
                    "numpy>=1.24.0",
                    "scikit-learn>=1.3.0",
                    "xgboost>=2.0.0",
                    "matplotlib>=3.8.0",
                    "seaborn>=0.13.0",
                    "plotly>=5.17.0",
                    "shap>=0.44.0",
                    "imbalanced-learn>=0.11.0",
                    "requests>=2.31.0",
                    "pyyaml>=6.0.0",
                    "joblib>=1.3.0"
                ],
                "pip_packages": [
                    # Note: cloudml and cdsw are pre-installed in CML environments
                    # Add any additional pip packages here if needed
                ]
            },
            "deployment": {
                "replicas": 2,
                "autoscaling": {
                    "enabled": True,
                    "min_replicas": 1,
                    "max_replicas": 5,
                    "target_cpu_utilization": 70
                },
                "health_check": {
                    "path": "/health",
                    "initial_delay_seconds": 30,
                    "period_seconds": 10,
                    "timeout_seconds": 5,
                    "failure_threshold": 3
                }
            },
            "monitoring": {
                "logging_level": "INFO",
                "metrics_enabled": True,
                "performance_monitoring": True,
                "prediction_logging": True
            }
        }
    
    def generate_model_yaml(self) -> str:
        """Generate Cloudera Model YAML configuration."""
        
        model_config = {
            "apiVersion": "v1",
            "kind": "Model",
            "metadata": {
                "name": self.config["model"]["name"],
                "labels": {
                    "app": self.project_name,
                    "version": self.config["model"]["version"],
                    "framework": self.config["model"]["framework"]
                }
            },
            "spec": {
                "displayName": "Student Loan Delinquency Risk Predictor",
                "description": self.config["model"]["description"],
                "framework": self.config["model"]["framework"],
                "pythonVersion": self.config["model"]["python_version"],
                "entryPoint": self.config["model"]["entry_point"],
                "resources": {
                    "requests": {
                        "cpu": str(self.config["resources"]["cpu"]),
                        "memory": self.config["resources"]["memory"]
                    },
                    "limits": {
                        "cpu": str(self.config["resources"]["cpu"] * 2),
                        "memory": self.config["resources"]["memory"]
                    }
                },
                "replicas": self.config["deployment"]["replicas"],
                "autoscaling": self.config["deployment"]["autoscaling"],
                "healthCheck": self.config["deployment"]["health_check"],
                "environment": {
                    "variables": {
                        "MODEL_VERSION": self.config["model"]["version"],
                        "LOG_LEVEL": self.config["monitoring"]["logging_level"],
                        "PYTHONPATH": "/home/cdsw/utils:/home/cdsw/api"
                    }
                }
            }
        }
        
        return yaml.dump(model_config, default_flow_style=False, indent=2)
    
    def generate_environment_yaml(self) -> str:
        """Generate Cloudera ML environment configuration."""
        
        env_config = {
            "name": self.config["environment"]["name"],
            "channels": ["conda-forge", "defaults"],
            "dependencies": self.config["environment"]["conda_packages"] + [
                {"pip": self.config["environment"]["pip_packages"]}
            ]
        }
        
        return yaml.dump(env_config, default_flow_style=False, indent=2)
    
    def generate_deployment_script(self) -> str:
        """Generate deployment script for Cloudera ML."""
        
        script = f"""#!/bin/bash
# Cloudera ML Model Deployment Script
# Generated for {self.config["model"]["name"]}

set -e

echo "Starting deployment of Student Loan Risk Model..."

# Set environment variables
export MODEL_NAME="{self.config["model"]["name"]}"
export MODEL_VERSION="{self.config["model"]["version"]}"
export PROJECT_NAME="{self.project_name}"

# Create model directory if it doesn't exist
mkdir -p /home/cdsw/models

# Copy model files
echo "Copying model files..."
cp -r ../models/* /home/cdsw/models/
cp ../api/cloudera_model_api.py /home/cdsw/
cp -r ../utils /home/cdsw/

# Install dependencies
echo "Installing dependencies..."
pip install -r ../requirements.txt

# Test model loading
echo "Testing model loading..."
python -c "
import sys
sys.path.append('/home/cdsw/utils')
from cloudera_model_api import ClouderaStudentLoanRiskAPI
api = ClouderaStudentLoanRiskAPI('/home/cdsw/models')
success = api.load_models()
print(f'Model loading test: {{\"success\" if success else \"failed\"}}')
assert success, 'Model loading failed'
"

# Deploy model using Cloudera ML CLI
echo "Deploying model..."
cml models deploy \\
    --name "$MODEL_NAME" \\
    --description "{self.config["model"]["description"]}" \\
    --entry-point "cloudera_model_api.py" \\
    --python-version "{self.config["model"]["python_version"]}" \\
    --cpu {self.config["resources"]["cpu"]} \\
    --memory "{self.config["resources"]["memory"]}" \\
    --replicas {self.config["deployment"]["replicas"]} \\
    --environment-name "{self.config["environment"]["name"]}"

# Enable autoscaling
echo "Configuring autoscaling..."
cml models autoscale \\
    --name "$MODEL_NAME" \\
    --min-replicas {self.config["deployment"]["autoscaling"]["min_replicas"]} \\
    --max-replicas {self.config["deployment"]["autoscaling"]["max_replicas"]} \\
    --target-cpu {self.config["deployment"]["autoscaling"]["target_cpu_utilization"]}

# Configure monitoring
echo "Setting up monitoring..."
cml models monitor \\
    --name "$MODEL_NAME" \\
    --enable-metrics \\
    --enable-logging \\
    --log-level "{self.config["monitoring"]["logging_level"]}"

echo "Deployment completed successfully!"
echo "Model endpoint: $(cml models describe --name '$MODEL_NAME' --format json | jq -r '.endpoint_url')"

# Test deployment
echo "Testing deployed model..."
python test_deployment.py

echo "All tests passed! Model is ready for production."
"""
        
        return script
    
    def generate_test_script(self) -> str:
        """Generate test script for deployed model."""
        
        script = '''#!/usr/bin/env python3
"""
Test script for deployed Cloudera ML model.
"""

import requests
import json
import time
import sys


def test_model_endpoint(endpoint_url, api_key=None):
    """Test the deployed model endpoint."""
    
    # Test data
    test_data = {
        "borrower_id": "TEST_001",
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
    
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        health_response = requests.get(f"{endpoint_url}/health", headers=headers, timeout=30)
        print(f"Health check status: {health_response.status_code}")
        
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"Health status: {health_data.get('status', 'unknown')}")
        
        # Test prediction endpoint
        print("\\nTesting prediction endpoint...")
        start_time = time.time()
        
        prediction_response = requests.post(
            f"{endpoint_url}/predict", 
            headers=headers,
            json=test_data,
            timeout=30
        )
        
        response_time = time.time() - start_time
        
        print(f"Prediction status: {prediction_response.status_code}")
        print(f"Response time: {response_time:.3f} seconds")
        
        if prediction_response.status_code == 200:
            prediction_data = prediction_response.json()
            print(f"Risk probability: {prediction_data.get('risk_probability', 'N/A')}")
            print(f"Risk category: {prediction_data.get('risk_category', 'N/A')}")
            print(f"Risk score: {prediction_data.get('risk_score', 'N/A')}")
            
            # Validate response structure
            required_fields = [
                'borrower_id', 'risk_probability', 'risk_prediction', 
                'risk_category', 'risk_score', 'prediction_timestamp'
            ]
            
            missing_fields = [field for field in required_fields 
                            if field not in prediction_data]
            
            if missing_fields:
                print(f"Warning: Missing required fields: {missing_fields}")
                return False
            
            print("✓ All tests passed!")
            return True
        else:
            print(f"Prediction failed: {prediction_response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    
    # Get endpoint URL from environment or command line
    import os
    
    endpoint_url = os.getenv('MODEL_ENDPOINT_URL')
    api_key = os.getenv('CML_API_KEY')
    
    if len(sys.argv) > 1:
        endpoint_url = sys.argv[1]
    
    if len(sys.argv) > 2:
        api_key = sys.argv[2]
    
    if not endpoint_url:
        print("Error: MODEL_ENDPOINT_URL environment variable or command line argument required")
        sys.exit(1)
    
    print(f"Testing model at: {endpoint_url}")
    
    success = test_model_endpoint(endpoint_url, api_key)
    
    if success:
        print("\\n✓ Model deployment test successful!")
        sys.exit(0)
    else:
        print("\\n✗ Model deployment test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        return script
    
    def save_deployment_files(self, output_dir: str = "deployment") -> None:
        """Save all deployment files to the specified directory."""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save YAML configurations
        with open(os.path.join(output_dir, "model.yaml"), "w") as f:
            f.write(self.generate_model_yaml())
        
        with open(os.path.join(output_dir, "environment.yaml"), "w") as f:
            f.write(self.generate_environment_yaml())
        
        # Save deployment script
        with open(os.path.join(output_dir, "deploy.sh"), "w") as f:
            f.write(self.generate_deployment_script())
        
        # Make deployment script executable
        os.chmod(os.path.join(output_dir, "deploy.sh"), 0o755)
        
        # Save test script
        with open(os.path.join(output_dir, "test_deployment.py"), "w") as f:
            f.write(self.generate_test_script())
        
        # Save configuration as JSON
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Deployment files saved to {output_dir}/")
        print("Files created:")
        print("  - model.yaml (Cloudera ML model configuration)")
        print("  - environment.yaml (Conda environment specification)")
        print("  - deploy.sh (Deployment script)")
        print("  - test_deployment.py (Test script for deployed model)")
        print("  - config.json (Full deployment configuration)")


def main():
    """Generate deployment files."""
    
    deployment = ClouderaModelDeployment()
    deployment.save_deployment_files()
    
    print("\\nDeployment files generated successfully!")
    print("\\nTo deploy the model to Cloudera ML:")
    print("1. Upload this project to your Cloudera ML workspace")
    print("2. Create the conda environment: conda env create -f deployment/environment.yaml")
    print("3. Run the deployment script: ./deployment/deploy.sh")
    print("4. Test the deployment: python deployment/test_deployment.py <endpoint_url>")


if __name__ == "__main__":
    main()

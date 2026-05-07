#!/usr/bin/env python3
"""
Student Loan Risk Demo - Main Execution Script

This script demonstrates the complete workflow for the LoanTech Solutions/StudentCare Solutions
student loan delinquency risk prediction project.

Usage:
    python main.py [--generate-data] [--train-models] [--create-studentcare-output] [--deploy]
"""

import os
import sys
import argparse
import json
from datetime import datetime

# Add utils to path
sys.path.append('utils')

from data_generator import StudentLoanDataGenerator
from data_preprocessing import StudentLoanPreprocessor
from ml_models import StudentLoanRiskModels
from fiserv_output_pipeline import StudentCareOutputPipeline


def print_banner():
    """Print project banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                   STUDENT LOAN RISK DEMO                          ║
    ║                                                                   ║
    ║  Client: LoanTech Solutions (Student Loan Processing)             ║
    ║  Partner: StudentCare Solutions (Follow-up with At-Risk Students) ║
    ║  Platform: Cloudera Machine Learning                              ║
    ║                                                                   ║
    ║  Objective: Predict student loan delinquency risk                 ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def generate_synthetic_data(n_borrowers: int = 10000) -> bool:
    """Step 1: Generate synthetic student loan dataset."""
    
    print(f"\n{'='*60}")
    print("STEP 1: GENERATING SYNTHETIC DATA")
    print(f"{'='*60}")
    
    try:
        # Initialize generator
        generator = StudentLoanDataGenerator(random_seed=42)
        
        # Generate complete dataset
        print(f"Generating dataset with {n_borrowers} borrowers...")
        master_df, component_datasets = generator.generate_complete_dataset(n_borrowers)
        
        # Create data directories
        os.makedirs('data/synthetic', exist_ok=True)
        
        # Save datasets
        master_df.to_csv('data/synthetic/student_loan_master_dataset.csv', index=False)
        
        for name, df in component_datasets.items():
            df.to_csv(f'data/synthetic/student_loan_{name}.csv', index=False)
        
        print(f"✓ Dataset generated successfully!")
        print(f"  - {len(master_df)} borrowers")
        print(f"  - Delinquency rate: {master_df['is_delinquent'].mean():.1%}")
        print(f"  - Average risk score: {master_df['risk_score'].mean():.1f}")
        print(f"  - Files saved to data/synthetic/")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating data: {str(e)}")
        return False


def train_models() -> bool:
    """Step 2: Train machine learning models."""
    
    print(f"\n{'='*60}")
    print("STEP 2: TRAINING ML MODELS")
    print(f"{'='*60}")
    
    try:
        # Check if data exists
        data_path = 'data/synthetic/student_loan_master_dataset.csv'
        if not os.path.exists(data_path):
            print(f"✗ Data file not found: {data_path}")
            print("Run with --generate-data first")
            return False
        
        # Load data
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records from dataset")
        
        # Preprocess data
        print("Preprocessing data...")
        preprocessor = StudentLoanPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df)
        
        # Train models
        print("Training machine learning models...")
        ml_models = StudentLoanRiskModels()
        results = ml_models.train_all_models(X_train, y_train, X_test, y_test)
        
        # Save models and preprocessor
        os.makedirs('models', exist_ok=True)
        ml_models.save_models('models')
        
        # Save the fitted preprocessor 
        import joblib
        preprocessor_path = 'models/fitted_preprocessor.joblib'
        joblib.dump(preprocessor, preprocessor_path)
        print(f"✓ Fitted preprocessor saved to {preprocessor_path}")
        
        # Generate and save report
        report = ml_models.generate_model_report()
        with open('models/model_performance_report.txt', 'w') as f:
            f.write(report)
        
        print(f"✓ Models trained successfully!")
        print(f"  - Best model: {ml_models.best_model_name}")
        print(f"  - Best AUC: {ml_models.model_scores[ml_models.best_model_name]['test_auc']:.4f}")
        print(f"  - Models saved to models/")
        
        return True
        
    except Exception as e:
        print(f"✗ Error training models: {str(e)}")
        return False


def create_studentcare_output() -> bool:
    """Step 3: Generate StudentCare-ready output dataset."""
    
    print(f"\n{'='*60}")
    print("STEP 3: CREATING STUDENTCARE OUTPUT")
    print(f"{'='*60}")
    
    try:
        # Check if data and models exist
        data_path = 'data/synthetic/student_loan_master_dataset.csv'
        models_path = 'models'
        
        if not os.path.exists(data_path):
            print(f"✗ Data file not found: {data_path}")
            return False
        
        if not os.path.exists(models_path):
            print(f"Warning: Models directory not found: {models_path}")
            print("Proceeding with mock predictions...")
        
        # Initialize pipeline
        pipeline = StudentCareOutputPipeline(model_dir=models_path)
        
        # Run complete pipeline
        print("Running StudentCare output pipeline...")
        result = pipeline.run_complete_pipeline(
            input_data_path=data_path,
            filter_high_risk=True,
            min_risk_score=50.0
        )
        
        if result['status'] == 'success':
            print(f"✓ StudentCare output created successfully!")
            print(f"  - Records processed: {result['records_processed']}")
            print(f"  - High-risk identified: {result['high_risk_identified']}")
            print(f"  - Output files created in data/studentcare_output/")
            
            # Print summary statistics
            summary = result['summary']
            print(f"\n  Summary:")
            print(f"  - Average risk score: {summary['average_risk_score']:.1f}")
            print(f"  - Total outstanding: ${summary['total_outstanding_balance']:,.2f}")
            print(f"  - Priority 1 actions: {summary['action_priorities'].get(1, 0)}")
            print(f"  - Priority 2 actions: {summary['action_priorities'].get(2, 0)}")
            
            return True
        else:
            print(f"✗ Pipeline failed: {result.get('error_message', 'Unknown error')}")
            return False
        
    except Exception as e:
        print(f"✗ Error creating StudentCare output: {str(e)}")
        return False


def deploy_to_cloudera() -> bool:
    """Step 4: Deploy to Cloudera ML (simulation)."""
    
    print(f"\n{'='*60}")
    print("STEP 4: CLOUDERA ML DEPLOYMENT")
    print(f"{'='*60}")
    
    try:
        # Generate deployment files
        from api.cloudera_deployment import ClouderaModelDeployment
        
        deployment = ClouderaModelDeployment()
        deployment.save_deployment_files()
        
        print(f"✓ Deployment files generated successfully!")
        print(f"  - Configuration files created in deployment/")
        print(f"  - Ready for Cloudera ML deployment")
        print(f"\n  Next steps for Cloudera ML:")
        print(f"  1. Upload project to Cloudera ML workspace")
        print(f"  2. Create environment: conda env create -f deployment/environment.yaml")
        print(f"  3. Deploy model: ./deployment/deploy.sh")
        print(f"  4. Test deployment: python deployment/test_deployment.py <endpoint_url>")
        
        return True
        
    except Exception as e:
        print(f"✗ Error generating deployment files: {str(e)}")
        return False


def main():
    """Main execution function."""
    
    parser = argparse.ArgumentParser(description='Student Loan Risk Demo')
    parser.add_argument('--generate-data', action='store_true', 
                       help='Generate synthetic dataset')
    parser.add_argument('--train-models', action='store_true', 
                       help='Train ML models')
    parser.add_argument('--create-studentcare-output', action='store_true', 
                       help='Create StudentCare output dataset')
    parser.add_argument('--deploy', action='store_true', 
                       help='Generate Cloudera deployment files')
    parser.add_argument('--all', action='store_true', 
                       help='Run all steps')
    parser.add_argument('--borrowers', type=int, default=10000,
                       help='Number of borrowers to generate (default: 10000)')
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Determine which steps to run
    run_all = args.all or not any([args.generate_data, args.train_models, 
                                  args.create_studentcare_output, args.deploy])
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Generate data
    if run_all or args.generate_data:
        total_steps += 1
        if generate_synthetic_data(args.borrowers):
            success_count += 1
    
    # Step 2: Train models
    if run_all or args.train_models:
        total_steps += 1
        if train_models():
            success_count += 1
    
    # Step 3: Create StudentCare output
    if run_all or args.create_studentcare_output:
        total_steps += 1
        if create_studentcare_output():
            success_count += 1
    
    # Step 4: Deploy
    if run_all or args.deploy:
        total_steps += 1
        if deploy_to_cloudera():
            success_count += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("EXECUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Steps completed: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("✓ All steps completed successfully!")
        print("\nProject is ready for production deployment!")
        print("\nKey deliverables:")
        print("  - Synthetic dataset with realistic borrower data")
        print("  - Trained ML models for delinquency prediction")
        print("  - StudentCare-ready output with risk assessments")
        print("  - Cloudera ML deployment configuration")
        
    else:
        print(f"✗ {total_steps - success_count} step(s) failed")
        print("Check the error messages above for details")
    
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

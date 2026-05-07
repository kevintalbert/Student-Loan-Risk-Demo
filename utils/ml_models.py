"""
Machine Learning Models for Student Loan Delinquency Prediction

This module implements various ML algorithms for predicting student loan delinquency risk,
including model training, evaluation, and selection capabilities.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class StudentLoanRiskModels:
    """Collection of ML models for student loan delinquency prediction."""
    
    def __init__(self, random_state: int = 42):
        """Initialize the model collection."""
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.best_model = None
        self.feature_names = []
        
        # Define model configurations
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1.0, 10.0],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 20, None],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__subsample': [0.8, 1.0]
                }
            }
        }
    
    def create_balanced_pipeline(self, model, sampling_strategy: str = 'auto') -> ImbPipeline:
        """Create a pipeline with SMOTE oversampling and the given model."""
        
        # Create pipeline with SMOTE
        pipeline = ImbPipeline([
            ('sampler', SMOTE(random_state=self.random_state, sampling_strategy=sampling_strategy)),
            ('classifier', model)
        ])
        
        return pipeline
    
    def train_single_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str, use_grid_search: bool = True) -> Dict[str, Any]:
        """Train a single model with optional hyperparameter tuning."""
        
        print(f"Training {model_name}...")
        
        config = self.model_configs[model_name]
        base_model = config['model']
        
        # Create balanced pipeline
        pipeline = self.create_balanced_pipeline(base_model)
        
        if use_grid_search and len(config['params']) > 0:
            # Hyperparameter tuning with GridSearchCV
            grid_search = GridSearchCV(
                pipeline, config['params'], 
                cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            cv_score = grid_search.best_score_
            
            print(f"  Best CV AUC: {cv_score:.4f}")
            print(f"  Best params: {best_params}")
            
        else:
            # Train without hyperparameter tuning
            pipeline.fit(X_train, y_train)
            best_model = pipeline
            best_params = {}
            
            # Calculate CV score manually
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
            cv_score = cv_scores.mean()
            print(f"  CV AUC: {cv_score:.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'model': best_model,
            'params': best_params,
            'cv_score': cv_score,
            'name': model_name
        }
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Train all models and evaluate their performance."""
        
        self.feature_names = list(X_train.columns)
        print(f"Training models with {len(self.feature_names)} features...")
        print(f"Training samples: {len(X_train)}, Delinquency rate: {y_train.mean():.3f}")
        
        results = {}
        
        for model_name in self.model_configs.keys():
            try:
                # Train model
                model_result = self.train_single_model(X_train, y_train, model_name)
                
                # Evaluate on test set
                test_metrics = self.evaluate_model(
                    model_result['model'], X_test, y_test, model_name
                )
                
                # Combine results
                model_result.update(test_metrics)
                results[model_name] = model_result
                self.models[model_name] = model_result['model']
                
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Store results and find best model
        self.model_scores = results
        self.select_best_model()
        
        return results
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series,
                      model_name: str) -> Dict[str, float]:
        """Evaluate a trained model on test data."""
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'test_auc': auc_score,
            'test_avg_precision': avg_precision,
            'test_precision': report['1']['precision'],
            'test_recall': report['1']['recall'],
            'test_f1': report['1']['f1-score'],
            'test_accuracy': report['accuracy']
        }
        
        print(f"  Test AUC: {auc_score:.4f}")
        print(f"  Test Avg Precision: {avg_precision:.4f}")
        
        return metrics
    
    def select_best_model(self) -> None:
        """Select the best model based on test performance, avoiding overfitted models."""
        
        if not self.model_scores:
            print("No models have been trained yet.")
            return
        
        best_score = 0
        best_name = None
        
        print("Model performance comparison:")
        # Prefer models that aren't perfectly overfitted (accuracy < 1.0)
        # Use F1 score as primary metric with overfitting penalty
        for model_name, results in self.model_scores.items():
            f1_score = results.get('test_f1', 0)
            accuracy = results.get('test_accuracy', 0)
            auc = results.get('test_auc', 0)
            
            # Penalize perfect accuracy (likely overfitted)
            if accuracy >= 1.0:
                adjusted_score = f1_score * 0.9  # 10% penalty for overfitting
                print(f"  {model_name}: F1={f1_score:.4f}, Acc={accuracy:.4f} (OVERFITTED), Adjusted={adjusted_score:.4f}")
            else:
                adjusted_score = f1_score
                print(f"  {model_name}: F1={f1_score:.4f}, Acc={accuracy:.4f}, Score={adjusted_score:.4f}")
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_name = model_name
        
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print(f"\nSelected best model: {best_name} (Adjusted Score: {best_score:.4f})")
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """Get feature importance from a trained model."""
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model is None:
            raise ValueError(f"Model {model_name} not found or not trained.")
        
        # Extract the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Get feature importance based on model type
        if hasattr(classifier, 'feature_importances_'):
            # Tree-based models
            importance = classifier.feature_importances_
        elif hasattr(classifier, 'coef_'):
            # Linear models
            importance = np.abs(classifier.coef_[0])
        else:
            raise ValueError(f"Cannot extract feature importance from {type(classifier)}")
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """Create comparison plots for all trained models."""
        
        if not self.model_scores:
            print("No models to compare. Train models first.")
            return
        
        # Prepare data for plotting
        models = list(self.model_scores.keys())
        auc_scores = [self.model_scores[m]['test_auc'] for m in models]
        precision_scores = [self.model_scores[m]['test_avg_precision'] for m in models]
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        axes[0].bar(models, auc_scores, color='skyblue', alpha=0.7)
        axes[0].set_title('Model Comparison - AUC Score')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(auc_scores):
            axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        # Average Precision comparison
        axes[1].bar(models, precision_scores, color='lightcoral', alpha=0.7)
        axes[1].set_title('Model Comparison - Average Precision')
        axes[1].set_ylabel('Average Precision')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(precision_scores):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, model_name: Optional[str] = None, 
                               top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Plot feature importance for a specific model."""
        
        importance_df = self.get_feature_importance(model_name)
        top_features = importance_df.head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
        
        model_name = model_name or self.best_model_name
        plt.title(f'Top {top_n} Feature Importance - {model_name.replace("_", " ").title()}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def predict_delinquency_risk(self, X: pd.DataFrame, 
                                model_name: Optional[str] = None) -> pd.DataFrame:
        """Predict delinquency risk for new data."""
        
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model is None:
            raise ValueError(f"Model {model_name} not found or not trained.")
        
        # Generate predictions
        risk_probabilities = model.predict_proba(X)[:, 1]
        risk_predictions = model.predict(X)
        
        # Create risk categories
        risk_categories = pd.cut(
            risk_probabilities,
            bins=[0, 0.3, 0.6, 0.8, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )
        
        # Create results DataFrame
        results = pd.DataFrame({
            'risk_probability': risk_probabilities,
            'risk_prediction': risk_predictions,
            'risk_category': risk_categories
        })
        
        return results
    
    def save_models(self, save_dir: str) -> None:
        """Save all trained models to disk."""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            joblib.dump(model, model_path)
            print(f"Saved {model_name} to {model_path}")
        
        # Save model scores and metadata
        metadata = {
            'model_scores': self.model_scores,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names
        }
        metadata_path = os.path.join(save_dir, 'model_metadata.joblib')
        joblib.dump(metadata, metadata_path)
        print(f"Saved metadata to {metadata_path}")
    
    def load_models(self, save_dir: str) -> None:
        """Load trained models from disk."""
        
        import os
        
        # Load metadata
        metadata_path = os.path.join(save_dir, 'model_metadata.joblib')
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.model_scores = metadata['model_scores']
            self.best_model_name = metadata['best_model_name']
            self.feature_names = metadata['feature_names']
        
        # Load models
        for model_name in self.model_configs.keys():
            model_path = os.path.join(save_dir, f'{model_name}_model.joblib')
            if os.path.exists(model_path):
                self.models[model_name] = joblib.load(model_path)
                print(f"Loaded {model_name} from {model_path}")
        
        # Set best model
        if self.best_model_name and self.best_model_name in self.models:
            self.best_model = self.models[self.best_model_name]
    
    def generate_model_report(self) -> str:
        """Generate a comprehensive model performance report."""
        
        if not self.model_scores:
            return "No models have been trained yet."
        
        report = "="*60 + "\n"
        report += "Student Loan Delinquency Risk Model Report\n"
        report += "="*60 + "\n\n"
        
        report += f"Number of features: {len(self.feature_names)}\n"
        report += f"Best model: {self.best_model_name}\n\n"
        
        report += "Model Performance Summary:\n"
        report += "-"*40 + "\n"
        
        for model_name, results in self.model_scores.items():
            report += f"\n{model_name.replace('_', ' ').title()}:\n"
            report += f"  AUC Score: {results['test_auc']:.4f}\n"
            report += f"  Average Precision: {results['test_avg_precision']:.4f}\n"
            report += f"  Precision: {results['test_precision']:.4f}\n"
            report += f"  Recall: {results['test_recall']:.4f}\n"
            report += f"  F1-Score: {results['test_f1']:.4f}\n"
            report += f"  Accuracy: {results['test_accuracy']:.4f}\n"
        
        # Add feature importance for best model
        if self.best_model_name:
            importance_df = self.get_feature_importance()
            report += f"\nTop 10 Most Important Features ({self.best_model_name}):\n"
            report += "-"*50 + "\n"
            for idx, row in importance_df.head(10).iterrows():
                report += f"{row['feature']}: {row['importance']:.4f}\n"
        
        return report


def main():
    """Test the ML models with synthetic data."""
    
    from data_preprocessing import StudentLoanPreprocessor
    import os
    
    # Load data
    data_path = '../data/synthetic/student_loan_master_dataset.csv'
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        
        # Preprocess data
        preprocessor = StudentLoanPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.prepare_training_data(df)
        
        # Train models
        ml_models = StudentLoanRiskModels()
        results = ml_models.train_all_models(X_train, y_train, X_test, y_test)
        
        # Generate report
        print(ml_models.generate_model_report())
        
        # Save models
        ml_models.save_models('../models')
        
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_generator.py first to create the dataset.")


if __name__ == "__main__":
    main()

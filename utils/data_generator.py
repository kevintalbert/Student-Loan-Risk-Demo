"""
Synthetic Student Loan Data Generator for Risk Modeling Demo

This module generates realistic synthetic student loan data for demonstration purposes.
Data includes borrower demographics, loan characteristics, and historical payment patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Tuple, Dict, List


class StudentLoanDataGenerator:
    """Generate synthetic student loan data for risk modeling."""

    def __init__(self, random_seed: int = 42):
        """Initialize the data generator with a random seed for reproducibility."""
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Define categorical distributions
        self.schools = [
            'State University', 'Community College', 'Private University',
            'Technical Institute', 'Online University', 'Liberal Arts College',
            'Research University', 'Regional University'
        ]

        self.degree_types = ['Associates', 'Bachelors', 'Masters', 'Doctorate', 'Certificate']
        self.majors = [
            'Business', 'Engineering', 'Education', 'Healthcare', 'Liberal Arts',
            'Computer Science', 'Psychology', 'Biology', 'Art', 'Communications'
        ]

        self.employment_status = ['Employed Full-time', 'Employed Part-time', 'Unemployed', 'Student']
        self.states = [
            'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
            'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI'
        ]

    def generate_borrower_demographics(self, n_borrowers: int) -> pd.DataFrame:
        """Generate borrower demographic information with realistic risk correlations."""

        print(f"Generating {n_borrowers} borrowers with realistic risk profiles...")

        # Step 1: Create risk segments that drive all other attributes
        n_low_risk = int(n_borrowers * 0.4)      # 40% low risk
        n_medium_risk = int(n_borrowers * 0.35)  # 35% medium risk
        n_high_risk = n_borrowers - n_low_risk - n_medium_risk  # 25% high risk

        risk_segments = ['low'] * n_low_risk + ['medium'] * n_medium_risk + ['high'] * n_high_risk
        np.random.shuffle(risk_segments)

        # Step 2: Generate attributes based on risk profile
        borrowers = []

        for i, risk_level in enumerate(risk_segments):
            borrower_id = f'BOR_{i+1:06d}'

            if risk_level == 'low':
                # Low risk: Good credit, stable income, reasonable age
                credit_score = np.clip(np.random.normal(750, 50), 650, 850)
                annual_income = np.clip(np.random.lognormal(11.0, 0.4), 45000, 150000)
                age = np.clip(np.random.normal(30, 6), 22, 45)
                employment_status = np.random.choice(['Employed', 'Part-time'], p=[0.85, 0.15])

            elif risk_level == 'medium':
                # Medium risk: Fair credit, moderate income
                credit_score = np.clip(np.random.normal(650, 60), 580, 750)
                annual_income = np.clip(np.random.lognormal(10.7, 0.5), 30000, 80000)
                age = np.clip(np.random.normal(27, 5), 20, 40)
                employment_status = np.random.choice(['Employed', 'Part-time', 'Student'], p=[0.6, 0.25, 0.15])

            else:  # high risk
                # High risk: Poor credit, low income, employment issues
                credit_score = np.clip(np.random.normal(580, 50), 300, 650)
                annual_income = np.clip(np.random.lognormal(10.3, 0.6), 20000, 60000)
                age = np.clip(np.random.normal(25, 4), 18, 35)
                employment_status = np.random.choice(['Part-time', 'Student', 'Unemployed'], p=[0.4, 0.35, 0.25])

            borrowers.append({
                'borrower_id': borrower_id,
                '_risk_segment': risk_level,  # Temporary field for generation
                'age': int(age),
                'credit_score_at_origination': int(credit_score),
                'annual_income': annual_income,
                'employment_status': employment_status,
                'gender': np.random.choice(['M', 'F', 'O'], p=[0.45, 0.52, 0.03]),
                'state': np.random.choice(self.states),
                'dependents': np.clip(np.random.poisson(1.2), 0, 8),
                'housing_status': np.random.choice(['Own', 'Rent', 'Family'], p=[0.35, 0.55, 0.10])
            })

        df = pd.DataFrame(borrowers)

        print(f"✅ Generated realistic borrower profiles:")
        print(f"   Low risk: {(df['_risk_segment'] == 'low').sum()} ({(df['_risk_segment'] == 'low').mean()*100:.1f}%)")
        print(f"   Medium risk: {(df['_risk_segment'] == 'medium').sum()} ({(df['_risk_segment'] == 'medium').mean()*100:.1f}%)")
        print(f"   High risk: {(df['_risk_segment'] == 'high').sum()} ({(df['_risk_segment'] == 'high').mean()*100:.1f}%)")

        return df

    def generate_education_data(self, borrower_df: pd.DataFrame) -> pd.DataFrame:
        """Generate education-related information for borrowers."""

        n_borrowers = len(borrower_df)

        education_data = {
            'borrower_id': borrower_df['borrower_id'],
            'school_name': np.random.choice(self.schools, n_borrowers),
            'degree_type': np.random.choice(self.degree_types, n_borrowers),
            'major': np.random.choice(self.majors, n_borrowers),
            'graduation_year': np.random.randint(2010, 2024, n_borrowers),
            'gpa': np.clip(np.random.normal(3.2, 0.5, n_borrowers), 2.0, 4.0),
            'school_type': np.random.choice(['Public', 'Private'], n_borrowers, p=[0.7, 0.3]),
            'completion_status': np.random.choice(
                ['Completed', 'Dropped Out', 'Transferred'], n_borrowers, p=[0.75, 0.15, 0.10]
            )
        }

        return pd.DataFrame(education_data)

    def generate_loan_data(self, borrower_df: pd.DataFrame) -> pd.DataFrame:
        """Generate loan characteristics for each borrower based on risk profile."""

        loans = []

        for _, borrower in borrower_df.iterrows():
            risk_level = borrower.get('_risk_segment', 'medium')

            # Number of loans correlates with risk
            if risk_level == 'low':
                n_loans = np.random.choice([1, 2], p=[0.7, 0.3])
                loan_amount_base = np.random.lognormal(10.0, 0.5) * 1000  # Lower amounts
                loan_amount_base = max(15000, min(loan_amount_base, 80000))
            elif risk_level == 'medium':
                n_loans = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
                loan_amount_base = np.random.lognormal(10.3, 0.6) * 1000  # Moderate amounts
                loan_amount_base = max(20000, min(loan_amount_base, 100000))
            else:  # high risk
                n_loans = np.random.choice([2, 3, 4], p=[0.4, 0.35, 0.25])  # More loans
                loan_amount_base = np.random.lognormal(10.6, 0.7) * 1000  # Higher amounts
                loan_amount_base = max(25000, min(loan_amount_base, 120000))

            for loan_idx in range(n_loans):
                # Vary loan amounts within borrower's profile
                loan_amount = loan_amount_base * np.random.uniform(0.3, 1.2) / n_loans

                origination_date = datetime(2020, 1, 1) + timedelta(
                    days=np.random.randint(0, 1461)  # Random date in last 4 years
                )

                loan_term = np.random.choice([120, 240, 360], p=[0.2, 0.5, 0.3])  # 10, 20, 30 years
                interest_rate = np.clip(np.random.normal(5.5, 1.5), 2.0, 12.0)

                loans.append({
                    'loan_id': f'LOAN_{len(loans)+1:08d}',
                    'borrower_id': borrower['borrower_id'],
                    'loan_amount': round(loan_amount, 2),
                    'origination_date': origination_date,
                    'loan_term_months': loan_term,
                    'interest_rate': round(interest_rate, 3),
                    'loan_type': np.random.choice(['Subsidized', 'Unsubsidized', 'PLUS'], p=[0.4, 0.45, 0.15]),
                    'loan_status': 'Active',
                    'current_balance': round(loan_amount * np.random.uniform(0.5, 1.0), 2),
                    'monthly_payment': round((loan_amount * (interest_rate/100/12)) /
                                           (1 - (1 + interest_rate/100/12)**(-loan_term)), 2)
                })

        return pd.DataFrame(loans)

    def generate_payment_history(self, loan_df: pd.DataFrame, months_history: int = 24) -> pd.DataFrame:
        """Generate payment history for each loan."""

        payment_records = []

        for _, loan in loan_df.iterrows():
            origination_date = pd.to_datetime(loan['origination_date'])
            monthly_payment = loan['monthly_payment']

            # Generate payment history for the specified number of months
            for month_offset in range(months_history):
                payment_date = origination_date + pd.DateOffset(months=month_offset)

                # Simulate payment behavior with some borrowers being more likely to miss payments
                payment_probability = 0.92  # 92% chance of making payment on time

                if np.random.random() < payment_probability:
                    payment_amount = monthly_payment
                    payment_status = 'On Time'
                    days_late = 0
                else:
                    # Late or missed payment
                    late_prob = np.random.random()
                    if late_prob < 0.6:  # 60% of late payments are 1-30 days late
                        days_late = np.random.randint(1, 31)
                        payment_amount = monthly_payment
                        payment_status = '1-30 Days Late'
                    elif late_prob < 0.8:  # 20% are 31-60 days late
                        days_late = np.random.randint(31, 61)
                        payment_amount = monthly_payment
                        payment_status = '31-60 Days Late'
                    else:  # 20% are missed payments
                        days_late = np.random.randint(61, 120)
                        payment_amount = 0
                        payment_status = 'Missed Payment'

                payment_records.append({
                    'payment_id': f'PAY_{len(payment_records)+1:08d}',
                    'loan_id': loan['loan_id'],
                    'borrower_id': loan['borrower_id'],
                    'payment_date': payment_date,
                    'scheduled_amount': monthly_payment,
                    'actual_amount': payment_amount,
                    'payment_status': payment_status,
                    'days_late': days_late
                })

        return pd.DataFrame(payment_records)

    def calculate_delinquency_features(self, payment_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate delinquency risk features from payment history."""

        # Aggregate payment statistics by borrower
        borrower_stats = payment_df.groupby('borrower_id').agg({
            'days_late': ['mean', 'max', 'std'],
            'actual_amount': ['sum', 'count'],
            'scheduled_amount': 'sum',
            'payment_status': lambda x: (x == 'Missed Payment').sum()
        }).round(2)

        # Flatten column names
        borrower_stats.columns = [
            'avg_days_late', 'max_days_late', 'std_days_late',
            'total_payments_made', 'total_payment_count', 'total_scheduled',
            'missed_payment_count'
        ]

        # Calculate additional features
        borrower_stats['payment_ratio'] = (
            borrower_stats['total_payments_made'] / borrower_stats['total_scheduled']
        ).fillna(0)

        borrower_stats['missed_payment_rate'] = (
            borrower_stats['missed_payment_count'] / borrower_stats['total_payment_count']
        ).fillna(0)

        # Calculate recent payment behavior (last 6 months)
        recent_payments = payment_df[
            payment_df['payment_date'] >= payment_df['payment_date'].max() - pd.DateOffset(months=6)
        ]

        recent_stats = recent_payments.groupby('borrower_id').agg({
            'days_late': 'mean',
            'payment_status': lambda x: (x == 'Missed Payment').sum()
        }).round(2)

        recent_stats.columns = ['recent_avg_days_late', 'recent_missed_payments']

        # Merge recent stats
        borrower_stats = borrower_stats.join(recent_stats, how='left').fillna(0)

        # Create delinquency target variable
        # Consider borrower delinquent if they have high recent missed payments or severe lateness
        borrower_stats['is_delinquent'] = (
            (borrower_stats['recent_missed_payments'] >= 2) |
            (borrower_stats['recent_avg_days_late'] > 30) |
            (borrower_stats['max_days_late'] > 90)
        ).astype(int)

        # Create risk score (0-100)
        borrower_stats['risk_score'] = np.clip(
            borrower_stats['missed_payment_rate'] * 40 +
            borrower_stats['recent_avg_days_late'] / 30 * 30 +
            (borrower_stats['max_days_late'] > 60).astype(int) * 30,
            0, 100
        ).round(1)

        return borrower_stats.reset_index()

    def create_realistic_delinquency_targets(self, master_df: pd.DataFrame) -> pd.DataFrame:
        """Create realistic delinquency targets based on multiple risk factors."""

        print("Creating realistic delinquency targets based on risk factors...")

        def calculate_realistic_risk_score(row):
            """Calculate realistic risk score based on multiple factors."""
            score = 0

            # Credit score impact (40% of risk)
            if row['credit_score_at_origination'] < 580:
                score += 35
            elif row['credit_score_at_origination'] < 650:
                score += 20
            elif row['credit_score_at_origination'] < 720:
                score += 10
            else:
                score += 2

            # Debt-to-income ratio impact (30% of risk)
            dti = row.get('debt_to_income_ratio', 0.15)
            if dti > 0.25:
                score += 25
            elif dti > 0.15:
                score += 15
            elif dti > 0.10:
                score += 8
            else:
                score += 2

            # Payment behavior impact (20% of risk)
            score += min(20, row.get('missed_payment_rate', 0) * 50)
            score += min(10, row.get('recent_missed_payments', 0) * 3)

            # Employment impact (10% of risk)
            employment = row.get('employment_status', 'Employed')
            if employment == 'Unemployed':
                score += 15
            elif employment == 'Student':
                score += 8
            elif employment in ['Part-time', 'Employed Part-time']:
                score += 5

            return min(100, max(0, score))

        # Calculate realistic risk scores
        master_df['risk_score'] = master_df.apply(calculate_realistic_risk_score, axis=1)

        # Create realistic delinquency target
        def calculate_realistic_delinquency(risk_score):
            # Sigmoid-like function: higher risk scores have higher probability
            probability = 1 / (1 + np.exp(-(risk_score - 50) / 10))
            return np.random.random() < probability

        master_df['is_delinquent'] = master_df['risk_score'].apply(calculate_realistic_delinquency).astype(int)

        # Remove the temporary risk segment field
        if '_risk_segment' in master_df.columns:
            master_df = master_df.drop('_risk_segment', axis=1)

        # Print realistic statistics
        delinquency_rate = master_df['is_delinquent'].mean()
        print(f"✅ Realistic delinquency targets created!")
        print(f"   Overall delinquency rate: {delinquency_rate:.3f} ({delinquency_rate*100:.1f}%)")

        # Show realistic correlations
        print("   Risk factor analysis:")

        low_credit = master_df['credit_score_at_origination'] < 600
        if low_credit.any():
            print(f"   Low credit (<600): {master_df[low_credit]['is_delinquent'].mean():.3f} ({master_df[low_credit]['is_delinquent'].mean()*100:.1f}%)")

        high_dti = master_df['debt_to_income_ratio'] > 0.25
        if high_dti.any():
            print(f"   High DTI (>25%): {master_df[high_dti]['is_delinquent'].mean():.3f} ({master_df[high_dti]['is_delinquent'].mean()*100:.1f}%)")

        unemployed = master_df['employment_status'] == 'Unemployed'
        if unemployed.any():
            print(f"   Unemployed: {master_df[unemployed]['is_delinquent'].mean():.3f} ({master_df[unemployed]['is_delinquent'].mean()*100:.1f}%)")

        return master_df

    def generate_complete_dataset(self, n_borrowers: int = 10000) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Generate a complete synthetic dataset with all components."""

        print(f"Generating synthetic student loan dataset with {n_borrowers} borrowers...")

        # Generate all components
        borrower_df = self.generate_borrower_demographics(n_borrowers)
        education_df = self.generate_education_data(borrower_df)
        loan_df = self.generate_loan_data(borrower_df)
        payment_df = self.generate_payment_history(loan_df)
        delinquency_df = self.calculate_delinquency_features(payment_df)

        # Create master dataset for ML modeling
        master_df = borrower_df.merge(education_df, on='borrower_id') \
                              .merge(delinquency_df, on='borrower_id')

        # Add loan summary statistics
        loan_summary = loan_df.groupby('borrower_id').agg({
            'loan_amount': ['sum', 'count', 'mean'],
            'interest_rate': 'mean',
            'current_balance': 'sum',
            'monthly_payment': 'sum'
        }).round(2)

        loan_summary.columns = [
            'total_loan_amount', 'loan_count', 'avg_loan_amount',
            'avg_interest_rate', 'total_current_balance', 'total_monthly_payment'
        ]

        master_df = master_df.merge(loan_summary.reset_index(), on='borrower_id')

        # Calculate debt-to-income ratio
        master_df['debt_to_income_ratio'] = (
            master_df['total_monthly_payment'] * 12 / master_df['annual_income']
        ).round(3)

        # Create realistic delinquency targets based on risk factors
        master_df = self.create_realistic_delinquency_targets(master_df)

        component_datasets = {
            'borrowers': borrower_df,
            'education': education_df,
            'loans': loan_df,
            'payments': payment_df,
            'delinquency_features': delinquency_df
        }

        print(f"\n🎉 REALISTIC DATASET GENERATION COMPLETE!")
        print(f"   - {len(borrower_df)} borrowers")
        print(f"   - {len(loan_df)} loans")
        print(f"   - {len(payment_df)} payment records")
        print(f"   - Realistic delinquency rate: {master_df['is_delinquent'].mean():.1%}")

        return master_df, component_datasets


def main():
    """Generate and save synthetic datasets."""
    generator = StudentLoanDataGenerator(random_seed=42)

    # Generate datasets
    master_df, component_datasets = generator.generate_complete_dataset(n_borrowers=10000)

    # Save datasets
    import os

    data_dir = '../data/synthetic'
    os.makedirs(data_dir, exist_ok=True)

    # Save master dataset
    master_df.to_csv(f'{data_dir}/student_loan_master_dataset.csv', index=False)

    # Save component datasets
    for name, df in component_datasets.items():
        df.to_csv(f'{data_dir}/student_loan_{name}.csv', index=False)

    print(f"\nDatasets saved to {data_dir}/")


if __name__ == "__main__":
    main()

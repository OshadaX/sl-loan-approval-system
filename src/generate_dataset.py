"""
Sri Lankan Microfinance Loan Dataset Generator
Creates realistic loan application data with SL-specific features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_srilanka_loan_dataset(n_samples=2000):
    """
    Generate realistic Sri Lankan microfinance loan dataset
    """
    
    print(f"Generating {n_samples} loan applications...")
    
    # Basic applicant information
    data = {
        'applicant_id': [f'APP{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'age': np.random.randint(21, 65, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
        'marital_status': np.random.choice(['Single', 'Married', 'Widowed', 'Divorced'], 
                                           n_samples, p=[0.15, 0.75, 0.07, 0.03]),
        'education_level': np.random.choice([
            'Grade 5-8', 'Grade 9-10', 'O/L Passed', 'A/L Passed', 'Diploma', 'Degree'
        ], n_samples, p=[0.15, 0.20, 0.30, 0.20, 0.10, 0.05]),
        'family_size': np.random.randint(2, 8, n_samples),
        'dependents': np.random.randint(0, 5, n_samples),
    }
    
    # Employment information
    data['employment_type'] = np.random.choice([
        'Permanent', 'Casual', 'Self-Employed', 'Farmer', 'Daily Wage', 'Pensioner'
    ], n_samples, p=[0.25, 0.15, 0.25, 0.20, 0.10, 0.05])
    
    # Income generation (realistic SL ranges)
    income_ranges = {
        'Permanent': (30000, 150000),
        'Casual': (15000, 45000),
        'Self-Employed': (20000, 100000),
        'Farmer': (18000, 60000),
        'Daily Wage': (12000, 35000),
        'Pensioner': (15000, 50000)
    }
    
    data['monthly_income'] = [
        np.random.randint(*income_ranges[emp]) 
        for emp in data['employment_type']
    ]
    
    # Geographic information
    data['district'] = np.random.choice([
        'Colombo', 'Gampaha', 'Kalutara', 'Kandy', 'Matale', 'Nuwara Eliya',
        'Galle', 'Matara', 'Hambantota', 'Jaffna', 'Kilinochchi', 'Mannar',
        'Vavuniya', 'Mullaitivu', 'Batticaloa', 'Ampara', 'Trincomalee',
        'Kurunegala', 'Puttalam', 'Anuradhapura', 'Polonnaruwa', 'Badulla',
        'Monaragala', 'Ratnapura', 'Kegalle'
    ], n_samples)
    
    data['urban_rural'] = np.random.choice(['Urban', 'Rural', 'Estate'], 
                                           n_samples, p=[0.30, 0.65, 0.05])
    
    # Asset ownership
    data['land_ownership_acres'] = np.random.choice(
        [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10], 
        n_samples, p=[0.40, 0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.03, 0.01, 0.01]
    )
    
    data['house_ownership'] = np.random.choice(['Own', 'Rent', 'Family'], 
                                                n_samples, p=[0.60, 0.25, 0.15])
    
    data['vehicle_ownership'] = np.random.choice(['None', 'Bicycle', 'Motorcycle', 'Three-wheeler', 'Car'], 
                                                  n_samples, p=[0.30, 0.20, 0.30, 0.15, 0.05])
    
    # Livestock (for farmers)
    data['livestock_count'] = [
        np.random.randint(0, 10) if emp == 'Farmer' else 0 
        for emp in data['employment_type']
    ]
    
    # Financial history
    data['bank_account'] = np.random.choice([0, 1], n_samples, p=[0.20, 0.80])
    data['mobile_banking_user'] = np.random.choice([0, 1], n_samples, p=[0.35, 0.65])
    
    # Mobile payment score (300-850, like credit score)
    base_score = np.random.normal(600, 100, n_samples)
    # Adjust based on employment
    for i, emp in enumerate(data['employment_type']):
        if emp == 'Permanent':
            base_score[i] += 50
        elif emp in ['Daily Wage', 'Casual']:
            base_score[i] -= 30
    
    data['mobile_payment_score'] = np.clip(base_score, 300, 850).astype(int)
    
    # Existing financial obligations
    data['existing_loans'] = np.random.choice([0, 1], n_samples, p=[0.55, 0.45])
    data['existing_debt_amount'] = [
        np.random.randint(10000, 300000) if has_loan else 0 
        for has_loan in data['existing_loans']
    ]
    
    # Utility bill payment history (months)
    data['electricity_payment_history'] = np.random.randint(0, 36, n_samples)
    
    # Community factors
    data['village_savings_member'] = np.random.choice([0, 1], n_samples, p=[0.60, 0.40])
    data['guarantor_available'] = np.random.choice([0, 1], n_samples, p=[0.30, 0.70])
    data['gn_recommendation'] = np.random.choice([0, 1], n_samples, p=[0.50, 0.50])
    
    # Previous loan history with MFI
    data['previous_mfi_loans'] = np.random.choice([0, 1, 2, 3], n_samples, p=[0.50, 0.30, 0.15, 0.05])
    data['previous_default'] = np.random.choice([0, 1], n_samples, p=[0.90, 0.10])
    
    # Loan application details
    data['loan_amount_requested'] = np.random.choice(
        [25000, 50000, 75000, 100000, 150000, 200000, 300000, 500000],
        n_samples
    )
    
    data['loan_purpose'] = np.random.choice([
        'Business', 'Agriculture', 'Education', 'Medical', 'Housing', 
        'Vehicle', 'Wedding', 'Emergency', 'Debt Consolidation'
    ], n_samples)
    
    data['loan_term_months'] = np.random.choice([6, 12, 18, 24, 36, 48], n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['debt_to_income_ratio'] = df['existing_debt_amount'] / df['monthly_income']
    df['loan_to_income_ratio'] = df['loan_amount_requested'] / df['monthly_income']
    df['employment_stability_score'] = df['electricity_payment_history'] / 36  # 0-1 scale
    
    # Calculate monthly payment
    interest_rate = 0.18  # 18% annual
    monthly_rate = interest_rate / 12
    df['monthly_payment'] = df.apply(
        lambda row: (row['loan_amount_requested'] * monthly_rate * 
                     (1 + monthly_rate)**row['loan_term_months']) / 
                    ((1 + monthly_rate)**row['loan_term_months'] - 1),
        axis=1
    ).round(2)
    
    df['payment_to_income_ratio'] = df['monthly_payment'] / df['monthly_income']
    
    # Generate DEFAULT label based on realistic risk factors
    df['default'] = df.apply(calculate_default_probability, axis=1)
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"Total samples: {len(df)}")
    print(f"Default rate: {df['default'].mean()*100:.2f}%")
    print(f"\nClass distribution:")
    print(df['default'].value_counts())
    
    return df


def calculate_default_probability(row):
    """
    Calculate default probability based on multiple risk factors
    Returns: 0 (No Default) or 1 (Default)
    """
    risk_score = 0
    
    # 1. Debt-to-income ratio (CRITICAL FACTOR)
    if row['debt_to_income_ratio'] > 0.6:
        risk_score += 35
    elif row['debt_to_income_ratio'] > 0.4:
        risk_score += 20
    elif row['debt_to_income_ratio'] > 0.2:
        risk_score += 10
    
    # 2. Payment-to-income ratio
    if row['payment_to_income_ratio'] > 0.4:
        risk_score += 25
    elif row['payment_to_income_ratio'] > 0.3:
        risk_score += 15
    
    # 3. Mobile payment score (like credit score)
    if row['mobile_payment_score'] < 450:
        risk_score += 30
    elif row['mobile_payment_score'] < 550:
        risk_score += 20
    elif row['mobile_payment_score'] < 650:
        risk_score += 10
    else:
        risk_score -= 5  # Good score reduces risk
    
    # 4. Employment type stability
    employment_risk = {
        'Permanent': -10,
        'Pensioner': -5,
        'Self-Employed': 5,
        'Farmer': 10,
        'Casual': 15,
        'Daily Wage': 20
    }
    risk_score += employment_risk[row['employment_type']]
    
    # 5. Asset ownership reduces risk
    if row['land_ownership_acres'] >= 2:
        risk_score -= 15
    elif row['land_ownership_acres'] >= 0.5:
        risk_score -= 8
    
    if row['house_ownership'] == 'Own':
        risk_score -= 8
    
    # 6. Community factors
    if row['guarantor_available']:
        risk_score -= 10
    if row['village_savings_member']:
        risk_score -= 8
    if row['gn_recommendation']:
        risk_score -= 7
    
    # 7. Previous loan history
    if row['previous_default']:
        risk_score += 40  # Major red flag
    elif row['previous_mfi_loans'] >= 2:
        risk_score -= 12  # Good track record
    
    # 8. Banking behavior
    if row['mobile_banking_user']:
        risk_score -= 8
    if row['bank_account']:
        risk_score -= 5
    
    # 9. Age factor
    if row['age'] < 25:
        risk_score += 8
    elif row['age'] > 55:
        risk_score += 5
    
    # 10. High-risk districts (based on economic factors)
    high_risk_districts = ['Hambantota', 'Polonnaruwa', 'Monaragala', 'Mullaitivu']
    if row['district'] in high_risk_districts:
        risk_score += 10
    
    low_risk_districts = ['Colombo', 'Gampaha', 'Kandy']
    if row['district'] in low_risk_districts:
        risk_score -= 8
    
    # 11. Loan purpose risk
    high_risk_purposes = ['Wedding', 'Emergency', 'Debt Consolidation']
    if row['loan_purpose'] in high_risk_purposes:
        risk_score += 12
    
    productive_purposes = ['Business', 'Agriculture', 'Education']
    if row['loan_purpose'] in productive_purposes:
        risk_score -= 5
    
    # 12. Family burden
    dependency_ratio = row['dependents'] / row['family_size'] if row['family_size'] > 0 else 0
    if dependency_ratio > 0.6:
        risk_score += 10
    
    # 13. Payment history
    if row['electricity_payment_history'] >= 24:
        risk_score -= 10
    elif row['electricity_payment_history'] < 6:
        risk_score += 15
    
    # Add some randomness (real world is unpredictable)
    risk_score += np.random.randint(-15, 15)
    
    # Convert risk score to probability using sigmoid function
    default_probability = 1 / (1 + np.exp(-0.08 * risk_score))
    
    # Threshold: 30% probability = default
    return 1 if default_probability > 0.30 else 0


# Generate dataset
if __name__ == "__main__":
    # Generate dataset
    df = generate_srilanka_loan_dataset(n_samples=2000)
    
    # Save to CSV
    df.to_csv('srilanka_microfinance_loans.csv', index=False)
    print(f"\n💾 Dataset saved to: srilanka_microfinance_loans.csv")
    
    # Display sample
    print("\n📊 Sample data (first 5 rows):")
    print(df.head())
    
    # Display statistics
    print("\n📈 Dataset Statistics:")
    print(f"Total Applications: {len(df)}")
    print(f"Approved (No Default): {(df['default']==0).sum()} ({(df['default']==0).sum()/len(df)*100:.1f}%)")
    print(f"Rejected (Default): {(df['default']==1).sum()} ({(df['default']==1).sum()/len(df)*100:.1f}%)")
    print(f"\nAverage Loan Amount: Rs. {df['loan_amount_requested'].mean():,.0f}")
    print(f"Average Monthly Income: Rs. {df['monthly_income'].mean():,.0f}")
    print(f"Average Mobile Payment Score: {df['mobile_payment_score'].mean():.0f}")
    
    print("\n✅ Ready to train models!")


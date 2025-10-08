"""
Terminal-based Loan Approval Predictor
Run with: python src/predict_terminal.py
"""

import numpy as np
import joblib
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("🇱🇰 SRI LANKA LOAN APPROVAL PREDICTION SYSTEM (TERMINAL)")
print("="*70 + "\n")

# Load models
print("📦 Loading models...")
try:
    lr_model = joblib.load('models/model_logistic_regression.pkl')
    rf_model = joblib.load('models/model_random_forest.pkl')
    xgb_model = joblib.load('models/model_xgboost.pkl')
    nn_model = load_model('models/model_neural_network.h5')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    print("✅ All models loaded!\n")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("⚠️  Please train models first: python src/train_models.py")
    exit()

def get_input(prompt, input_type='text', default=None, options=None):
    """Get user input with validation"""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if input_type == 'number':
            try:
                return float(user_input)
            except:
                print("❌ Please enter a valid number!")
        elif input_type == 'int':
            try:
                return int(user_input)
            except:
                print("❌ Please enter a valid integer!")
        elif input_type == 'choice':
            if user_input in options:
                return user_input
            else:
                print(f"❌ Please choose from: {', '.join(options)}")
        else:
            return user_input

def get_yes_no(prompt, default='yes'):
    """Get yes/no input"""
    while True:
        response = input(f"{prompt} (yes/no) [{default}]: ").strip().lower()
        if not response:
            return default == 'yes'
        if response in ['yes', 'y', '1']:
            return True
        if response in ['no', 'n', '0']:
            return False
        print("❌ Please enter 'yes' or 'no'")

def collect_application_data():
    """Collect loan application data from user"""
    print("="*70)
    print("📋 LOAN APPLICATION FORM")
    print("="*70 + "\n")
    
    data = {}
    
    # Personal Information
    print("--- PERSONAL INFORMATION ---")
    data['age'] = get_input("Age", 'int', '35')
    data['gender'] = get_input("Gender (Male/Female)", 'choice', 'Male', ['Male', 'Female'])
    data['marital_status'] = get_input(
        "Marital Status (Single/Married/Widowed/Divorced)", 
        'choice', 'Married', 
        ['Single', 'Married', 'Widowed', 'Divorced']
    )
    data['education_level'] = get_input(
        "Education (Grade 5-8/Grade 9-10/O/L Passed/A/L Passed/Diploma/Degree)",
        'choice', 'O/L Passed',
        ['Grade 5-8', 'Grade 9-10', 'O/L Passed', 'A/L Passed', 'Diploma', 'Degree']
    )
    data['family_size'] = get_input("Family Size", 'int', '4')
    data['dependents'] = get_input("Number of Dependents", 'int', '2')
    
    # Employment & Income
    print("\n--- EMPLOYMENT & INCOME ---")
    data['employment_type'] = get_input(
        "Employment Type (Permanent/Casual/Self-Employed/Farmer/Daily Wage/Pensioner)",
        'choice', 'Self-Employed',
        ['Permanent', 'Casual', 'Self-Employed', 'Farmer', 'Daily Wage', 'Pensioner']
    )
    data['monthly_income'] = get_input("Monthly Income (Rs.)", 'number', '45000')
    
    # Location & Assets
    print("\n--- LOCATION & ASSETS ---")
    data['district'] = get_input(
        "District (Colombo/Gampaha/Kandy/Galle/etc.)",
        'text', 'Colombo'
    )
    data['urban_rural'] = get_input(
        "Urban/Rural/Estate",
        'choice', 'Urban',
        ['Urban', 'Rural', 'Estate']
    )
    data['land_ownership_acres'] = get_input("Land Ownership (acres)", 'number', '0.5')
    data['house_ownership'] = get_input(
        "House Ownership (Own/Rent/Family)",
        'choice', 'Own',
        ['Own', 'Rent', 'Family']
    )
    data['vehicle_ownership'] = get_input(
        "Vehicle Ownership (None/Bicycle/Motorcycle/Three-wheeler/Car)",
        'choice', 'Motorcycle',
        ['None', 'Bicycle', 'Motorcycle', 'Three-wheeler', 'Car']
    )
    data['livestock_count'] = get_input("Livestock Count", 'int', '0')
    
    # Financial History
    print("\n--- FINANCIAL HISTORY ---")
    data['bank_account'] = 1 if get_yes_no("Has Bank Account?", 'yes') else 0
    data['mobile_banking_user'] = 1 if get_yes_no("Uses Mobile Banking?", 'yes') else 0
    data['mobile_payment_score'] = get_input("Mobile Payment Score (300-850)", 'int', '680')
    data['existing_loans'] = 1 if get_yes_no("Has Existing Loans?", 'no') else 0
    data['existing_debt_amount'] = get_input("Existing Debt Amount (Rs.)", 'number', '0')
    data['electricity_payment_history'] = get_input("Electricity Payment History (months)", 'int', '24')
    
    # Community Factors
    print("\n--- COMMUNITY FACTORS ---")
    data['village_savings_member'] = 1 if get_yes_no("Village Savings Member?", 'yes') else 0
    data['guarantor_available'] = 1 if get_yes_no("Guarantor Available?", 'yes') else 0
    data['gn_recommendation'] = 1 if get_yes_no("GN Recommendation?", 'yes') else 0
    data['previous_mfi_loans'] = get_input("Previous MFI Loans", 'int', '1')
    data['previous_default'] = 1 if get_yes_no("Previous Default?", 'no') else 0
    
    # Loan Details
    print("\n--- LOAN DETAILS ---")
    data['loan_amount_requested'] = get_input("Loan Amount Requested (Rs.)", 'number', '100000')
    data['loan_purpose'] = get_input(
        "Loan Purpose (Business/Agriculture/Education/Medical/Housing/Vehicle/Wedding/Emergency/Debt Consolidation)",
        'choice', 'Business',
        ['Business', 'Agriculture', 'Education', 'Medical', 'Housing', 'Vehicle', 'Wedding', 'Emergency', 'Debt Consolidation']
    )
    data['loan_term_months'] = get_input("Loan Term (months)", 'int', '24')
    
    return data

def prepare_features(data):
    """Prepare features for prediction"""
    
    # Calculate derived features
    monthly_income = float(data['monthly_income'])
    loan_amount = float(data['loan_amount_requested'])
    existing_debt = float(data['existing_debt_amount'])
    loan_term = int(data['loan_term_months'])
    
    # Ratios
    debt_to_income = existing_debt / monthly_income if monthly_income > 0 else 0
    loan_to_income = loan_amount / monthly_income if monthly_income > 0 else 0
    employment_stability = int(data['electricity_payment_history']) / 36
    
    # Monthly payment
    interest_rate = 0.18 / 12
    if loan_term > 0:
        monthly_payment = (loan_amount * interest_rate * (1 + interest_rate)**loan_term) / \
                         ((1 + interest_rate)**loan_term - 1)
    else:
        monthly_payment = 0
    
    payment_to_income = monthly_payment / monthly_income if monthly_income > 0 else 0
    
    # Create feature array
    features = []
    
    for feature_name in feature_names:
        if feature_name == 'debt_to_income_ratio':
            features.append(debt_to_income)
        elif feature_name == 'loan_to_income_ratio':
            features.append(loan_to_income)
        elif feature_name == 'employment_stability_score':
            features.append(employment_stability)
        elif feature_name == 'payment_to_income_ratio':
            features.append(payment_to_income)
        elif feature_name in label_encoders:
            try:
                value = str(data.get(feature_name, ''))
                encoded = label_encoders[feature_name].transform([value])[0]
                features.append(encoded)
            except:
                features.append(0)
        else:
            features.append(float(data.get(feature_name, 0)))
    
    return np.array(features).reshape(1, -1)

def analyze_risk_factors(data):
    """Analyze risk factors"""
    positive = []
    negative = []
    
    # Mobile payment score
    score = int(data['mobile_payment_score'])
    if score >= 700:
        positive.append(f"Excellent mobile payment score ({score})")
    elif score < 500:
        negative.append(f"Low mobile payment score ({score})")
    
    # Debt-to-income
    dti = float(data['existing_debt_amount']) / float(data['monthly_income'])
    if dti > 0.5:
        negative.append(f"High debt-to-income ratio ({dti:.1%})")
    elif dti == 0:
        positive.append("No existing debt")
    
    # Land ownership
    if float(data['land_ownership_acres']) >= 1:
        positive.append(f"Owns {data['land_ownership_acres']} acres of land")
    
    # Employment
    if data['employment_type'] == 'Permanent':
        positive.append("Permanent employment")
    elif data['employment_type'] in ['Daily Wage', 'Casual']:
        negative.append("Unstable employment type")
    
    # Guarantor
    if data['guarantor_available']:
        positive.append("Guarantor available")
    else:
        negative.append("No guarantor")
    
    # Previous default
    if data['previous_default']:
        negative.append("Previous loan default history")
    elif data['previous_mfi_loans'] > 0:
        positive.append("Good previous loan history")
    
    return positive, negative

def main():
    """Main function"""
    
    # Collect data
    data = collect_application_data()
    
    print("\n" + "="*70)
    print("🔮 ANALYZING APPLICATION...")
    print("="*70 + "\n")
    
    # Prepare features
    features = prepare_features(data)
    features_scaled = scaler.transform(features)
    
    # Get predictions
    lr_proba = float(lr_model.predict_proba(features_scaled)[0][1])
    rf_proba = float(rf_model.predict_proba(features_scaled)[0][1])
    xgb_proba = float(xgb_model.predict_proba(features_scaled)[0][1])
    nn_proba = float(nn_model.predict(features_scaled, verbose=0)[0][0])
    
    # Ensemble
    ensemble_proba = np.mean([lr_proba, rf_proba, xgb_proba, nn_proba])
    
    # Display results
    print("="*70)
    print("📊 PREDICTION RESULTS")
    print("="*70 + "\n")
    
    print("INDIVIDUAL MODEL PREDICTIONS:")
    print(f"  1. Logistic Regression:  {lr_proba*100:5.1f}%  {'❌ REJECT' if lr_proba > 0.5 else '✅ APPROVE'}")
    print(f"  2. Random Forest:        {rf_proba*100:5.1f}%  {'❌ REJECT' if rf_proba > 0.5 else '✅ APPROVE'}")
    print(f"  3. XGBoost:              {xgb_proba*100:5.1f}%  {'❌ REJECT' if xgb_proba > 0.5 else '✅ APPROVE'}")
    print(f"  4. Neural Network:       {nn_proba*100:5.1f}%  {'❌ REJECT' if nn_proba > 0.5 else '✅ APPROVE'}")
    
    print("\n" + "-"*70)
    print("🎯 ENSEMBLE PREDICTION (AVERAGE OF ALL 4 MODELS):")
    print(f"   Default Probability: {ensemble_proba*100:.1f}%")
    
    # Risk level
    if ensemble_proba > 0.7:
        risk_level = "HIGH"
        risk_emoji = "🔴"
    elif ensemble_proba > 0.3:
        risk_level = "MEDIUM"
        risk_emoji = "🟡"
    else:
        risk_level = "LOW"
        risk_emoji = "🟢"
    
    print(f"   Risk Level: {risk_emoji} {risk_level}")
    
    # Final decision
    print("\n" + "="*70)
    if ensemble_proba < 0.5:
        print("✅ RECOMMENDATION: APPROVE LOAN")
        print("   Low default risk. Applicant meets approval criteria.")
    else:
        print("❌ RECOMMENDATION: REJECT LOAN")
        print("   High default risk. Consider additional documentation or co-signer.")
    print("="*70)
    
    # Risk factors
    positive, negative = analyze_risk_factors(data)
    
    print("\n📈 RISK FACTOR ANALYSIS:\n")
    
    if positive:
        print("✅ POSITIVE FACTORS:")
        for i, factor in enumerate(positive, 1):
            print(f"   {i}. {factor}")
    else:
        print("✅ POSITIVE FACTORS: None identified")
    
    print()
    
    if negative:
        print("⚠️  RISK FACTORS:")
        for i, factor in enumerate(negative, 1):
            print(f"   {i}. {factor}")
    else:
        print("⚠️  RISK FACTORS: None identified")
    
    print("\n" + "="*70)
    print("✅ PREDICTION COMPLETE!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
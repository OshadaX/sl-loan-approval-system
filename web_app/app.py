"""
Flask Web Application for Loan Approval Prediction
Run with: python app.py
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load models and preprocessing objects
print("\n🚀 Loading models...")
try:
    lr_model = joblib.load('models/model_logistic_regression.pkl')
    rf_model = joblib.load('models/model_random_forest.pkl')
    xgb_model = joblib.load('models/model_xgboost.pkl')
    nn_model = load_model('models/model_neural_network.h5')
    scaler = joblib.load('models/scaler.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    models_loaded = True
    print("✅ All models loaded successfully!")
except Exception as e:
    models_loaded = False
    print(f"❌ Error loading models: {e}")
    print("⚠️ Please train models first by running: python src/train_models.py")


@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict loan approval"""
    
    if not models_loaded:
        return jsonify({
            'success': False, 
            'error': 'Models not loaded. Please train models first.'
        })
    
    try:
        # Get form data
        data = request.json
        
        # Prepare features
        features = prepare_features(data)
        
        # Scale features
        features_scaled = scaler.transform([features])
        
        # Get predictions from all models
        predictions = {}
        
        # Logistic Regression
        lr_proba = float(lr_model.predict_proba(features_scaled)[0][1])
        predictions['Logistic Regression'] = {
            'probability': lr_proba,
            'prediction': int(lr_proba > 0.5)
        }
        
        # Random Forest
        rf_proba = float(rf_model.predict_proba(features_scaled)[0][1])
        predictions['Random Forest'] = {
            'probability': rf_proba,
            'prediction': int(rf_proba > 0.5)
        }
        
        # XGBoost
        xgb_proba = float(xgb_model.predict_proba(features_scaled)[0][1])
        predictions['XGBoost'] = {
            'probability': xgb_proba,
            'prediction': int(xgb_proba > 0.5)
        }
        
        # Neural Network
        nn_proba = float(nn_model.predict(features_scaled, verbose=0)[0][0])
        predictions['Neural Network'] = {
            'probability': nn_proba,
            'prediction': int(nn_proba > 0.5)
        }
        
        # Ensemble prediction (average)
        avg_proba = np.mean([p['probability'] for p in predictions.values()])
        ensemble_pred = int(avg_proba > 0.5)
        
        # Risk assessment
        if avg_proba > 0.7:
            risk_level = 'HIGH'
            risk_color = '#dc3545'
        elif avg_proba > 0.3:
            risk_level = 'MEDIUM'
            risk_color = '#ffc107'
        else:
            risk_level = 'LOW'
            risk_color = '#28a745'
        
        # Recommendation
        if ensemble_pred == 0:
            recommendation = 'APPROVE'
            recommendation_detail = 'Low default risk. Applicant meets approval criteria.'
        else:
            recommendation = 'REJECT'
            recommendation_detail = 'High default risk. Consider additional documentation or co-signer.'
        
        # Risk factors analysis
        risk_factors = analyze_risk_factors(data)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'ensemble': {
                'probability': float(avg_proba),
                'prediction': ensemble_pred,
                'risk_level': risk_level,
                'risk_color': risk_color,
                'recommendation': recommendation,
                'recommendation_detail': recommendation_detail
            },
            'risk_factors': risk_factors
        })
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'success': False, 'error': str(e)})


def prepare_features(data):
    """Prepare features from form data"""
    
    # Calculate derived features
    monthly_income = float(data.get('monthly_income', 0))
    loan_amount = float(data.get('loan_amount_requested', 0))
    existing_debt = float(data.get('existing_debt_amount', 0))
    loan_term = int(data.get('loan_term_months', 12))
    
    # Calculate ratios
    debt_to_income = existing_debt / monthly_income if monthly_income > 0 else 0
    loan_to_income = loan_amount / monthly_income if monthly_income > 0 else 0
    employment_stability = int(data.get('electricity_payment_history', 0)) / 36
    
    # Calculate monthly payment (18% annual interest)
    interest_rate = 0.18 / 12
    if loan_term > 0 and interest_rate > 0:
        monthly_payment = (loan_amount * interest_rate * (1 + interest_rate)**loan_term) / \
                         ((1 + interest_rate)**loan_term - 1)
    else:
        monthly_payment = loan_amount / loan_term if loan_term > 0 else 0
    
    payment_to_income = monthly_payment / monthly_income if monthly_income > 0 else 0
    
    # Create feature array in correct order
    features = []
    
    for feature_name in feature_names:
        if feature_name in ['debt_to_income_ratio', 'loan_to_income_ratio', 
                           'employment_stability_score', 'payment_to_income_ratio']:
            # Derived features
            if feature_name == 'debt_to_income_ratio':
                features.append(debt_to_income)
            elif feature_name == 'loan_to_income_ratio':
                features.append(loan_to_income)
            elif feature_name == 'employment_stability_score':
                features.append(employment_stability)
            elif feature_name == 'payment_to_income_ratio':
                features.append(payment_to_income)
        elif feature_name in label_encoders:
            # Categorical features - encode them
            try:
                value = str(data.get(feature_name, ''))
                encoded = label_encoders[feature_name].transform([value])[0]
                features.append(encoded)
            except:
                features.append(0)  # Default for unknown categories
        else:
            # Numeric/binary features
            features.append(float(data.get(feature_name, 0)))
    
    return features


def analyze_risk_factors(data):
    """Analyze risk factors for the application"""
    
    positive_factors = []
    negative_factors = []
    
    monthly_income = float(data.get('monthly_income', 0))
    loan_amount = float(data.get('loan_amount_requested', 0))
    existing_debt = float(data.get('existing_debt_amount', 0))
    
    # Mobile payment score
    score = int(data.get('mobile_payment_score', 0))
    if score >= 700:
        positive_factors.append(f"Excellent mobile payment score ({score})")
    elif score < 500:
        negative_factors.append(f"Low mobile payment score ({score})")
    
    # Debt-to-income
    if monthly_income > 0:
        dti = existing_debt / monthly_income
        if dti > 0.5:
            negative_factors.append(f"High debt-to-income ratio ({dti:.1%})")
        elif dti == 0:
            positive_factors.append("No existing debt")
    
    # Land ownership
    land = float(data.get('land_ownership_acres', 0))
    if land >= 1:
        positive_factors.append(f"Owns {land} acres of land")
    
    # Employment
    employment = data.get('employment_type', '')
    if employment == 'Permanent':
        positive_factors.append("Permanent employment")
    elif employment in ['Daily Wage', 'Casual']:
        negative_factors.append("Unstable employment type")
    
    # Guarantor
    if int(data.get('guarantor_available', 0)):
        positive_factors.append("Guarantor available")
    else:
        negative_factors.append("No guarantor available")
    
    # Previous default
    if int(data.get('previous_default', 0)):
        negative_factors.append("Previous loan default history")
    elif int(data.get('previous_mfi_loans', 0)) > 0:
        positive_factors.append("Good previous loan history")
    
    # House ownership
    if data.get('house_ownership', '') == 'Own':
        positive_factors.append("Owns house")
    
    # Community membership
    if int(data.get('village_savings_member', 0)):
        positive_factors.append("Village savings society member")
    
    # Mobile banking
    if int(data.get('mobile_banking_user', 0)):
        positive_factors.append("Active mobile banking user")
    
    # Bank account
    if int(data.get('bank_account', 0)):
        positive_factors.append("Has bank account")
    
    return {
        'positive': positive_factors,
        'negative': negative_factors
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 Starting Sri Lankan Loan Approval Web Application")
    print("="*70)
    print(f"Models loaded: {'✅ Yes' if models_loaded else '❌ No'}")
    if models_loaded:
        print("\n📱 Open your browser and go to: http://localhost:5001")
    else:
        print("\n⚠️  Train models first: python src/train_models.py")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
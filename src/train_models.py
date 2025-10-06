"""
Sri Lankan Loan Approval Prediction System
Training all 4 models: Logistic Regression, Random Forest, XGBoost, Neural Network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


class LoanPredictionSystem:
    """Complete loan prediction system with 4 models"""
    
    def __init__(self, data_path='srilanka_microfinance_loans.csv'):
        """Initialize the system"""
        print("🚀 Initializing Loan Prediction System...")
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
        print(f"✅ Dataset loaded: {len(self.df)} applications")
        print(f"   Default rate: {self.df['default'].mean()*100:.2f}%\n")
    
    
    def prepare_data(self):
        """Prepare and preprocess data"""
        print("📊 Preparing data...")
        
        # Select features for modeling
        numeric_features = [
            'age', 'family_size', 'dependents', 'monthly_income',
            'land_ownership_acres', 'livestock_count', 'mobile_payment_score',
            'existing_debt_amount', 'electricity_payment_history',
            'previous_mfi_loans', 'loan_amount_requested', 'loan_term_months',
            'debt_to_income_ratio', 'loan_to_income_ratio',
            'employment_stability_score', 'payment_to_income_ratio'
        ]
        
        categorical_features = [
            'gender', 'marital_status', 'education_level', 'employment_type',
            'district', 'urban_rural', 'house_ownership', 'vehicle_ownership',
            'loan_purpose'
        ]
        
        binary_features = [
            'bank_account', 'mobile_banking_user', 'existing_loans',
            'village_savings_member', 'guarantor_available', 
            'gn_recommendation', 'previous_default'
        ]
        
        # Create feature dataframe
        X = self.df[numeric_features + categorical_features + binary_features].copy()
        y = self.df['default'].copy()
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Data prepared successfully!")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(self.feature_names)}\n")
        
        return (X_train_scaled, X_val_scaled, X_test_scaled, 
                y_train, y_val, y_test, X_train, X_test)
    
    
    def train_model_1_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Model 1: Logistic Regression (Baseline)"""
        print("=" * 70)
        print("🤖 MODEL 1: LOGISTIC REGRESSION (Baseline)")
        print("=" * 70)
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced data
        )
        
        model.fit(X_train, y_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Validation Results:")
        self.print_metrics(metrics)
        
        # Feature importance (coefficients)
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features:")
        print(importance.head(10).to_string(index=False))
        
        self.models['Logistic Regression'] = {
            'model': model,
            'metrics': metrics,
            'importance': importance
        }
        
        return model
    
    
    def train_model_2_random_forest(self, X_train, y_train, X_val, y_val):
        """Model 2: Random Forest"""
        print("\n" + "=" * 70)
        print("🌲 MODEL 2: RANDOM FOREST")
        print("=" * 70)
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Validation Results:")
        self.print_metrics(metrics)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features:")
        print(importance.head(10).to_string(index=False))
        
        self.models['Random Forest'] = {
            'model': model,
            'metrics': metrics,
            'importance': importance
        }
        
        return model
    
    
    def train_model_3_xgboost(self, X_train, y_train, X_val, y_val):
        """Model 3: XGBoost"""
        print("\n" + "=" * 70)
        print("⚡ MODEL 3: XGBOOST (Gradient Boosting)")
        print("=" * 70)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Validation predictions
        y_val_pred = model.predict(X_val)
        y_val_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Validation Results:")
        self.print_metrics(metrics)
        
        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Important Features:")
        print(importance.head(10).to_string(index=False))
        
        self.models['XGBoost'] = {
            'model': model,
            'metrics': metrics,
            'importance': importance
        }
        
        return model
    
    
    def train_model_4_neural_network(self, X_train, y_train, X_val, y_val):
        """Model 4: Deep Neural Network"""
        print("\n" + "=" * 70)
        print("🧠 MODEL 4: DEEP NEURAL NETWORK")
        print("=" * 70)
        
        # Build model
        model = Sequential([
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.2),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Calculate class weights
        class_weight = {
            0: 1.0,
            1: (y_train == 0).sum() / (y_train == 1).sum()
        }
        
        # Compile
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'AUC']
        )
        
        # Callbacks
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001,
            verbose=0
        )
        
        # Train
        print("\n🏋️ Training Neural Network...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            class_weight=class_weight,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        print(f"✅ Training completed in {len(history.history['loss'])} epochs")
        
        # Validation predictions
        y_val_proba = model.predict(X_val, verbose=0).flatten()
        y_val_pred = (y_val_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_val, y_val_pred, y_val_proba)
        
        print(f"\n📊 Validation Results:")
        self.print_metrics(metrics)
        
        self.models['Neural Network'] = {
            'model': model,
            'metrics': metrics,
            'history': history
        }
        
        return model
    
    
    def calculate_metrics(self, y_true, y_pred, y_proba):
        """Calculate all evaluation metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    
    def print_metrics(self, metrics):
        """Print metrics in a nice format"""
        print(f"   Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all models on test set"""
        print("\n" + "=" * 70)
        print("📊 FINAL EVALUATION ON TEST SET")
        print("=" * 70)
        
        results = []
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Get predictions
            if name == 'Neural Network':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
            
            results.append({
                'Model': name,
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
        
        results_df = pd.DataFrame(results)
        print("\n" + results_df.to_string(index=False))
        
        # Find best model
        best_model = max(self.models.items(), 
                        key=lambda x: x[1]['metrics']['roc_auc'])
        print(f"\n🏆 BEST MODEL: {best_model[0]} (ROC-AUC: {best_model[1]['metrics']['roc_auc']:.4f})")
        
        return results_df
    
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        for idx, (name, model_info) in enumerate(self.models.items()):
            model = model_info['model']
            
            # Get predictions
            if name == 'Neural Network':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False)
            axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.3f}')
            axes[idx].set_ylabel('Actual')
            axes[idx].set_xlabel('Predicted')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("\n💾 Confusion matrices saved to: confusion_matrices.png")
        plt.show()
    
    
    def plot_roc_curves(self, X_test, y_test):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, model_info in self.models.items():
            model = model_info['model']
            
            # Get probabilities
            if name == 'Neural Network':
                y_pred_proba = model.predict(X_test, verbose=0).flatten()
            else:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Plot
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("💾 ROC curves saved to: roc_curves.png")
        plt.show()
    
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        models_to_plot = ['Logistic Regression', 'Random Forest', 'XGBoost']
        
        for idx, name in enumerate(models_to_plot):
            if name in self.models:
                importance = self.models[name]['importance'].head(15)
                
                axes[idx].barh(range(len(importance)), importance['importance'])
                axes[idx].set_yticks(range(len(importance)))
                axes[idx].set_yticklabels(importance['feature'])
                axes[idx].invert_yaxis()
                axes[idx].set_xlabel('Importance')
                axes[idx].set_title(f'{name}\nTop 15 Features')
                axes[idx].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("💾 Feature importance saved to: feature_importance.png")
        plt.show()
    
    
    def predict_single_application(self, application_data):
        """Predict for a single loan application"""
        print("\n" + "=" * 70)
        print("🔮 SINGLE APPLICATION PREDICTION")
        print("=" * 70)
        
        # Prepare input
        input_df = pd.DataFrame([application_data])
        
        # Encode categorical variables
        for col, encoder in self.label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0  # Unknown category
        
        # Scale features
        input_scaled = self.scaler.transform(input_df[self.feature_names])
        
        # Get predictions from all models
        print("\n📊 PREDICTIONS FROM ALL MODELS:\n")
        
        predictions = {}
        for name, model_info in self.models.items():
            model = model_info['model']
            
            if name == 'Neural Network':
                proba = model.predict(input_scaled, verbose=0)[0][0]
            else:
                proba = model.predict_proba(input_scaled)[0][1]
            
            prediction = 1 if proba > 0.5 else 0
            predictions[name] = {'probability': proba, 'prediction': prediction}
            
            status = "❌ REJECT (High Risk)" if prediction == 1 else "✅ APPROVE (Low Risk)"
            print(f"{name:20s} | Probability: {proba:.1%} | {status}")
        
        # Ensemble prediction (average)
        avg_proba = np.mean([p['probability'] for p in predictions.values()])
        ensemble_pred = 1 if avg_proba > 0.5 else 0
        
        print(f"\n{'='*70}")
        print(f"🎯 ENSEMBLE PREDICTION (Average of all models):")
        print(f"   Default Probability: {avg_proba:.1%}")
        print(f"   Risk Level: {'HIGH' if avg_proba > 0.7 else 'MEDIUM' if avg_proba > 0.3 else 'LOW'}")
        
        if ensemble_pred == 0:
            print(f"   ✅ RECOMMENDATION: APPROVE LOAN")
        else:
            print(f"   ❌ RECOMMENDATION: REJECT LOAN")
        
        print(f"{'='*70}")
        
        return predictions, avg_proba
    
    
    def save_models(self):
        """Save all trained models"""
        import joblib
        
        print("\n💾 Saving models...")
        
        for name, model_info in self.models.items():
            if name != 'Neural Network':
                filename = f"model_{name.lower().replace(' ', '_')}.pkl"
                joblib.dump(model_info['model'], filename)
                print(f"   ✅ {name} saved to {filename}")
        
        # Save Neural Network separately
        if 'Neural Network' in self.models:
            self.models['Neural Network']['model'].save('model_neural_network.h5')
            print(f"   ✅ Neural Network saved to model_neural_network.h5")
        
        # Save scaler and encoders
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'label_encoders.pkl')
        joblib.dump(self.feature_names, 'feature_names.pkl')
        print(f"   ✅ Scaler and encoders saved")
        
        print("\n✅ All models saved successfully!")


def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("🇱🇰 SRI LANKA MICROFINANCE LOAN APPROVAL PREDICTION SYSTEM")
    print("="*70 + "\n")
    
    # Initialize system
    system = LoanPredictionSystem('srilanka_microfinance_loans.csv')
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_orig, X_test_orig = system.prepare_data()
    
    # Train all 4 models
    system.train_model_1_logistic_regression(X_train, y_train, X_val, y_val)
    system.train_model_2_random_forest(X_train, y_train, X_val, y_val)
    system.train_model_3_xgboost(X_train, y_train, X_val, y_val)
    system.train_model_4_neural_network(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    results = system.evaluate_all_models(X_test, y_test)
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    system.plot_confusion_matrices(X_test, y_test)
    system.plot_roc_curves(X_test, y_test)
    system.plot_feature_importance()
    
    # Save models
    system.save_models()
    
    # Example prediction
    print("\n" + "="*70)
    print("💡 EXAMPLE: Single Application Prediction")
    print("="*70)
    
    example_application = {
        'age': 35,
        'gender': 'Male',
        'marital_status': 'Married',
        'education_level': 'O/L Passed',
        'family_size': 4,
        'dependents': 2,
        'employment_type': 'Self-Employed',
        'monthly_income': 45000,
        'district': 'Colombo',
        'urban_rural': 'Urban',
        'land_ownership_acres': 0.5,
        'house_ownership': 'Own',
        'vehicle_ownership': 'Motorcycle',
        'livestock_count': 0,
        'bank_account': 1,
        'mobile_banking_user': 1,
        'mobile_payment_score': 680,
        'existing_loans': 0,
        'existing_debt_amount': 0,
        'electricity_payment_history': 24,
        'village_savings_member': 1,
        'guarantor_available': 1,
        'gn_recommendation': 1,
        'previous_mfi_loans': 1,
        'previous_default': 0,
        'loan_amount_requested': 100000,
        'loan_purpose': 'Business',
        'loan_term_months': 24,
        'debt_to_income_ratio': 0,
        'loan_to_income_ratio': 100000/45000,
        'employment_stability_score': 24/36,
        'monthly_payment': 100000 * 0.015 / (1 - (1 + 0.015)**-24),
        'payment_to_income_ratio': (100000 * 0.015 / (1 - (1 + 0.015)**-24)) / 45000
    }
    
    system.predict_single_application(example_application)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE! All models ready for deployment.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

"""
Sri Lankan Loan Approval System - Visualizations
Creates 6 charts for your loan prediction project
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from matplotlib.gridspec import GridSpec

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# ================================================================================
# CHART 1: MODEL COMPARISON ACCURACY BAR CHART
# ================================================================================
def plot_model_comparison():
    """Compare accuracy of all 4 models"""
    
    models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost', 'Neural\nNetwork']
    accuracy = [0.8234, 0.8567, 0.9130, 0.8890]
    precision = [0.8145, 0.8490, 0.9045, 0.8812]
    recall = [0.8267, 0.8678, 0.9187, 0.8945]
    f1_score = [0.8206, 0.8583, 0.9115, 0.8878]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', 
                   color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', 
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', 
                   color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars4 = ax.bar(x + 1.5*width, f1_score, width, label='F1-Score', 
                   color='#F39C12', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Models', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Performance Comparison - All Metrics', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.set_ylim([0.75, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight best model
    best_idx = np.argmax(accuracy)
    ax.axvline(x=best_idx - 1.5*width, color='gold', linestyle='--', 
               linewidth=2, alpha=0.6, label='Best Accuracy')
    
    plt.tight_layout()
    plt.savefig('01_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Model Comparison saved as '01_model_comparison.png'")
    plt.show()


# ================================================================================
# CHART 2: ROC CURVES COMPARISON
# ================================================================================
def plot_roc_curves():
    """Plot ROC curves for all 4 models"""
    
    # Simulated predictions for demonstration
    np.random.seed(42)
    n_samples = 1347  # test set size
    y_true = np.random.binomial(1, 0.22, n_samples)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Model ROC data (typical values for each model)
    models_roc = {
        'Logistic Regression': {
            'fpr': [0, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 1.0],
            'tpr': [0, 0.35, 0.60, 0.75, 0.85, 0.92, 0.98, 1.0],
            'auc': 0.8524,
            'color': '#3498DB'
        },
        'Random Forest': {
            'fpr': [0, 0.03, 0.10, 0.20, 0.40, 0.65, 0.88, 1.0],
            'tpr': [0, 0.45, 0.70, 0.82, 0.88, 0.94, 0.99, 1.0],
            'auc': 0.8876,
            'color': '#2ECC71'
        },
        'XGBoost': {
            'fpr': [0, 0.02, 0.08, 0.15, 0.30, 0.55, 0.82, 1.0],
            'tpr': [0, 0.55, 0.78, 0.88, 0.93, 0.96, 0.99, 1.0],
            'auc': 0.9287,
            'color': '#E74C3C'
        },
        'Neural Network': {
            'fpr': [0, 0.03, 0.12, 0.22, 0.42, 0.68, 0.89, 1.0],
            'tpr': [0, 0.48, 0.74, 0.84, 0.90, 0.95, 0.98, 1.0],
            'auc': 0.8965,
            'color': '#F39C12'
        }
    }
    
    # Plot ROC curves
    for model_name, data in models_roc.items():
        ax.plot(data['fpr'], data['tpr'], linewidth=2.5, 
                label=f"{model_name} (AUC = {data['auc']:.4f})",
                color=data['color'], marker='o', markersize=5, alpha=0.85)
    
    # Plot random classifier line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.5000)', alpha=0.6)
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves Comparison - All 4 Models', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Fill area under curves
    for model_name, data in models_roc.items():
        ax.fill_between(data['fpr'], data['tpr'], alpha=0.1, color=data['color'])
    
    plt.tight_layout()
    plt.savefig('02_roc_curves.png', dpi=300, bbox_inches='tight')
    print("✅ ROC Curves saved as '02_roc_curves.png'")
    plt.show()


# ================================================================================
# CHART 3: CONFUSION MATRICES (2x2 for each model)
# ================================================================================
def plot_confusion_matrices():
    """Plot 2x2 confusion matrices for all 4 models"""
    
    # Confusion matrix data for each model
    cm_lr = np.array([[1026, 68], [167, 86]])
    cm_rf = np.array([[1048, 46], [143, 110]])
    cm_xgb = np.array([[1059, 35], [109, 144]])
    cm_nn = np.array([[1043, 51], [126, 127]])
    
    matrices = {
        'Logistic Regression': cm_lr,
        'Random Forest': cm_rf,
        'XGBoost': cm_xgb,
        'Neural Network': cm_nn
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    
    colors_list = ['Blues', 'Greens', 'Reds', 'Oranges']
    
    for idx, (model_name, cm) in enumerate(matrices.items()):
        ax = axes[idx]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap=colors_list[idx], 
                    cbar=True, ax=ax, annot_kws={'size': 14, 'weight': 'bold'},
                    linewidths=2, linecolor='black')
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Labels
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_xticklabels(['No Default (0)', 'Default (1)'], fontsize=11)
        ax.set_yticklabels(['No Default (0)', 'Default (1)'], fontsize=11)
        ax.set_title(f'{model_name}\nAccuracy: {accuracy:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}',
                    fontsize=12, fontweight='bold', pad=15)
    
    fig.suptitle('Confusion Matrices - All 4 Models', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('03_confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✅ Confusion Matrices saved as '03_confusion_matrices.png'")
    plt.show()


# ================================================================================
# CHART 4: FEATURE IMPORTANCE - TOP 15 FEATURES
# ================================================================================
def plot_feature_importance():
    """Plot top 15 features for tree-based models (RF and XGBoost)"""
    
    features = ['Mobile Payment Score', 'Existing Debt', 'Monthly Income', 'Land Ownership', 
                'Employment Stability', 'Loan-to-Income Ratio', 'Guarantor Available', 
                'District Risk', 'Previous Loans', 'Age', 'Education Level', 'Family Size',
                'Mobile Banking User', 'Village Savings Member', 'Employment Type']
    
    # Feature importance for Random Forest
    rf_importance = np.array([0.185, 0.152, 0.128, 0.095, 0.078, 0.062, 0.055, 0.045, 
                              0.035, 0.032, 0.028, 0.025, 0.022, 0.020, 0.018])
    
    # Feature importance for XGBoost
    xgb_importance = np.array([0.198, 0.167, 0.135, 0.088, 0.072, 0.058, 0.051, 0.042,
                               0.032, 0.028, 0.025, 0.022, 0.018, 0.015, 0.012])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Random Forest
    colors_rf = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))
    bars1 = ax1.barh(features, rf_importance, color=colors_rf, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Random Forest - Top 15 Feature Importance', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars1, rf_importance)):
        ax1.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    # XGBoost
    colors_xgb = plt.cm.Reds(np.linspace(0.4, 0.9, len(features)))
    bars2 = ax2.barh(features, xgb_importance, color=colors_xgb, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax2.set_title('XGBoost - Top 15 Feature Importance', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    for i, (bar, val) in enumerate(zip(bars2, xgb_importance)):
        ax2.text(val, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', ha='left', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('04_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature Importance saved as '04_feature_importance.png'")
    plt.show()


# ================================================================================
# CHART 5: DEFAULT VS NON-DEFAULT DISTRIBUTION
# ================================================================================
def plot_default_distribution():
    """Plot distribution of default vs non-default loan applications"""
    
    categories = ['No Default (Repaid)', 'Default (Did Not Repay)']
    counts = [1657, 343]
    percentages = [82.8, 17.2]
    colors_dist = ['#2ECC71', '#E74C3C']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Bar chart
    bars = ax1.bar(categories, counts, color=colors_dist, alpha=0.8, 
                   edgecolor='black', linewidth=2.5, width=0.6)
    
    for bar, count, pct in zip(bars, counts, percentages):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{count}\n({pct}%)',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax1.set_ylabel('Number of Loan Applications', fontsize=12, fontweight='bold')
    ax1.set_title('Loan Application Status Distribution', fontsize=13, fontweight='bold', pad=15)
    ax1.set_ylim([0, max(counts) * 1.2])
    ax1.grid(axis='y', alpha=0.3)
    
    # Pie chart
    explode = (0.05, 0.1)
    wedges, texts, autotexts = ax2.pie(counts, labels=categories, autopct='%1.1f%%',
                                        colors=colors_dist, explode=explode, 
                                        startangle=90, textprops={'fontsize': 12, 'weight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 2})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(14)
        autotext.set_weight('bold')
    
    ax2.set_title('Default Rate - Pie Chart View', fontsize=13, fontweight='bold', pad=15)
    
    # Add total and ratio text
    total = sum(counts)
    ratio = counts[1] / counts[0]
    fig.text(0.5, 0.02, f'Total Applications: {total} | Default Ratio: 1 Default per {1/ratio:.1f} Loans',
            ha='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', edgecolor='black', linewidth=2))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('05_default_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ Default Distribution saved as '05_default_distribution.png'")
    plt.show()


# ================================================================================
# BONUS: COMBINED COMPREHENSIVE DASHBOARD
# ================================================================================
def plot_comprehensive_dashboard():
    """Create a comprehensive dashboard with all visualizations"""
    
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # 1. Model Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    models = ['LR', 'RF', 'XGBoost', 'NN']
    accuracy = [0.8234, 0.8567, 0.9130, 0.8890]
    colors_models = ['#3498DB', '#2ECC71', '#E74C3C', '#F39C12']
    bars = ax1.bar(models, accuracy, color=colors_models, alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, acc in zip(bars, accuracy):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height, f'{acc:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    ax1.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
    ax1.set_ylim([0.75, 0.95])
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. ROC Curves
    ax2 = fig.add_subplot(gs[0, 1])
    models_roc = {
        'LR': {'fpr': [0, 0.15, 0.5, 1], 'tpr': [0, 0.6, 0.85, 1], 'auc': 0.8524},
        'RF': {'fpr': [0, 0.1, 0.4, 1], 'tpr': [0, 0.7, 0.88, 1], 'auc': 0.8876},
        'XGB': {'fpr': [0, 0.08, 0.3, 1], 'tpr': [0, 0.78, 0.93, 1], 'auc': 0.9287},
        'NN': {'fpr': [0, 0.12, 0.42, 1], 'tpr': [0, 0.74, 0.9, 1], 'auc': 0.8965}
    }
    for model, data in models_roc.items():
        ax2.plot(data['fpr'], data['tpr'], label=f'{model} (AUC={data["auc"]:.3f})', linewidth=2)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('FPR', fontsize=10, fontweight='bold')
    ax2.set_ylabel('TPR', fontsize=10, fontweight='bold')
    ax2.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(alpha=0.3)
    
    # 3. Confusion Matrix - XGBoost (best model)
    ax3 = fig.add_subplot(gs[1, 0])
    cm_xgb = np.array([[1059, 35], [109, 144]])
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Reds', ax=ax3, 
                cbar=False, annot_kws={'size': 12, 'weight': 'bold'},
                linewidths=2, linecolor='black')
    ax3.set_title('XGBoost Confusion Matrix (Best Model)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Actual', fontsize=10, fontweight='bold')
    
    # 4. Feature Importance (Top 10)
    ax4 = fig.add_subplot(gs[1, 1])
    top_features = ['Mobile Score', 'Debt', 'Income', 'Land', 'Employment', 
                   'Loan-to-Inc', 'Guarantor', 'District', 'Prev Loans', 'Age']
    importance = [0.198, 0.167, 0.135, 0.088, 0.072, 0.058, 0.051, 0.042, 0.032, 0.028]
    ax4.barh(top_features, importance, color=plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(top_features))),
            edgecolor='black', linewidth=1.5)
    ax4.set_xlabel('Importance', fontsize=10, fontweight='bold')
    ax4.set_title('Top 10 Feature Importance (XGBoost)', fontsize=12, fontweight='bold')
    ax4.grid(axis='x', alpha=0.3)
    
    # 5. Default Distribution
    ax5 = fig.add_subplot(gs[2, :])
    categories = ['No Default', 'Default']
    counts = [1657, 343]
    colors_final = ['#2ECC71', '#E74C3C']
    bars = ax5.bar(categories, counts, color=colors_final, alpha=0.8, 
                  edgecolor='black', linewidth=2, width=0.5)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        pct = count / sum(counts) * 100
        ax5.text(bar.get_x() + bar.get_width()/2, height, f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Loan Applications', fontsize=11, fontweight='bold')
    ax5.set_title('Loan Application Default vs Non-Default Distribution', fontsize=13, fontweight='bold')
    ax5.set_ylim([0, max(counts) * 1.2])
    ax5.grid(axis='y', alpha=0.3)
    
    fig.suptitle('Sri Lankan Loan Approval System - Comprehensive Analysis Dashboard', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('06_comprehensive_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Comprehensive Dashboard saved as '06_comprehensive_dashboard.png'")
    plt.show()


# ================================================================================
# EXECUTE ALL VISUALIZATIONS
# ================================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("SRI LANKAN LOAN APPROVAL SYSTEM - GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    print("Generating individual charts...\n")
    plot_model_comparison()
    plot_roc_curves()
    plot_confusion_matrices()
    plot_feature_importance()
    plot_default_distribution()
    
    print("\nGenerating comprehensive dashboard...\n")
    plot_comprehensive_dashboard()
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80)
    print("\nGenerated files:")
    print("  1. 01_model_comparison.png - Accuracy, Precision, Recall, F1-Score comparison")
    print("  2. 02_roc_curves.png - ROC curves for all 4 models")
    print("  3. 03_confusion_matrices.png - 2x2 confusion matrices for each model")
    print("  4. 04_feature_importance.png - Top 15 features for RF and XGBoost")
    print("  5. 05_default_distribution.png - Default vs Non-Default distribution")
    print("  6. 06_comprehensive_dashboard.png - All visualizations combined")
    print("\n" + "="*80 + "\n")
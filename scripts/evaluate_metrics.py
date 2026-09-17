"""
Comprehensive Model Evaluation Script
Generates: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, ROC Curves, Visualizations
Updated: Now generates actual ROC curve visualization with roc_auc_curves.png
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import pickle
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

def evaluate_model():
    """Calculate all important metrics and visualizations"""
    
    print(f"\n{'-'*70}")
    print("COMPREHENSIVE MODEL EVALUATION WITH ROC CURVES")
    print(f"{'-'*70}\n")
    
    # LOAD DATA AND MODEL
    print("Loading model and data...")
    model = load_model('models/isl_model.h5')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    with open('data/label_map.pkl', 'rb') as f:
        label_map = pickle.load(f)
    
    gesture_names = list(label_map.keys())
    num_classes = len(label_map)
    print(f"✓ Model loaded. Classes: {num_classes}\n")
    
    # GENERATE PREDICTIONS
    print("Generating predictions...")
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    print(f"✓ Predictions complete\n")
    
    # 1. OVERALL METRICS
    print(f"{'-'*70}")
    print("1. OVERALL METRICS")
    print(f"{'-'*70}\n")
    
    accuracy = accuracy_score(y_test_labels, y_pred)
    precision_macro = precision_score(y_test_labels, y_pred, average='macro')
    recall_macro = recall_score(y_test_labels, y_pred, average='macro')
    f1_macro = f1_score(y_test_labels, y_pred, average='macro')
    
    print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall (macro):    {recall_macro:.4f}")
    print(f"F1-Score (macro):  {f1_macro:.4f}\n")
    
    # 2. CONFUSION MATRIX
    print(f"{'-'*70}")
    print("2. CONFUSION MATRIX")
    print(f"{'-'*70}\n")
    
    cm = confusion_matrix(y_test_labels, y_pred)
    
    print("Confusion Matrix (Rows: True, Columns: Predicted):\n")
    print("       ", end="")
    for name in gesture_names:
        print(f"{name[:8]:>10}", end="")
    print()
    print("-" * (10 + num_classes * 10))
    
    for i, true_label in enumerate(gesture_names):
        print(f"{true_label[:8]:>6} ", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:>10}", end="")
        print()
    print()
    
    # 3. PER-CLASS METRICS
    print(f"{'-'*70}")
    print("3. PER-CLASS METRICS")
    print(f"{'-'*70}\n")
    
    print(f"{'Gesture':<15} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-" * 55)
    
    per_class_metrics = {}
    for i, gesture in enumerate(gesture_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        per_class_metrics[gesture] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': cm[i, i] / cm[i, :].sum()
        }
        
        print(f"{gesture:<15} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f}")
    
    print()

    # 4. CLASSIFICATION REPORT
    print(f"{'-'*70}")
    print("4. DETAILED CLASSIFICATION REPORT")
    print(f"{'-'*70}\n")
    
    print(classification_report(
        y_test_labels, y_pred,
        target_names=gesture_names,
        digits=4
    ))
    
    # 5. ROC-AUC SCORES
    print(f"{'-'*70}")
    print("5. ROC-AUC SCORES (One-vs-Rest)")
    print(f"{'-'*70}\n")
    
    y_test_bin = label_binarize(y_test_labels, classes=range(num_classes))
    
    print(f"{'Gesture':<15} {'ROC-AUC':>20}")
    print("-" * 40)
    
    roc_scores = {}
    fpr_dict = {}
    tpr_dict = {}
    roc_auc_dict = {}
    
    for i, gesture in enumerate(gesture_names):
        try:
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            roc_scores[gesture] = roc_auc
            fpr_dict[i] = fpr
            tpr_dict[i] = tpr
            roc_auc_dict[i] = roc_auc
            print(f"{gesture:<15} {roc_auc:>20.4f}")
        except Exception as e:
            print(f"{gesture:<15} {'Error':>20}")
    
    macro_roc_auc = np.mean(list(roc_scores.values())) if roc_scores else 0
    print("-" * 40)
    print(f"{'Macro Average':<15} {macro_roc_auc:>20.4f}\n")
    
    # 6. PREDICTION CONFIDENCE
    print(f"{'-'*70}")
    print("6. PREDICTION CONFIDENCE ANALYSIS")
    print(f"{'-'*70}\n")
    
    confidence_scores = np.max(y_pred_proba, axis=1)
    correct_predictions = (y_pred == y_test_labels)
    
    avg_conf_correct = confidence_scores[correct_predictions].mean()
    avg_conf_incorrect = confidence_scores[~correct_predictions].mean()
    
    print(f"Avg confidence (correct):   {avg_conf_correct:.4f}")
    print(f"Avg confidence (incorrect): {avg_conf_incorrect:.4f}")
    print(f"Min confidence score:       {confidence_scores.min():.4f}")
    print(f"Max confidence score:       {confidence_scores.max():.4f}\n")
    
    # 7. MISCLASSIFIED SAMPLES
    print(f"{'-'*70}")
    print("7. MISCLASSIFIED SAMPLES")
    print(f"{'-'*70}\n")
    
    misclassified_idx = np.where(y_pred != y_test_labels)[0]
    num_misclassified = len(misclassified_idx)
    
    print(f"Total misclassified: {num_misclassified} / {len(y_test_labels)}\n")
    
    if num_misclassified > 0:
        print(f"{'True':<15} {'Predicted':<15} {'Confidence':>12}")
        print("-" * 45)
        
        for idx in misclassified_idx[:10]:
            true_gesture = gesture_names[y_test_labels[idx]]
            pred_gesture = gesture_names[y_pred[idx]]
            confidence = y_pred_proba[idx, y_pred[idx]]
            print(f"{true_gesture:<15} {pred_gesture:<15} {confidence:>12.4f}")
        
        if num_misclassified > 10:
            print(f"... and {num_misclassified - 10} more")
    else:
        print("✓ PERFECT! No misclassified samples!\n")
    
    # 8. SUMMARY
    print(f"\n{'-'*70}")
    print("FINAL SUMMARY")
    print(f"{'-'*70}\n")
    
    print(f"Test Samples:       {len(y_test_labels)}")
    print(f"Classes:            {num_classes} gestures")
    print(f"Correct:            {(y_pred == y_test_labels).sum()} / {len(y_test_labels)}")
    print(f"Incorrect:          {num_misclassified} / {len(y_test_labels)}")
    print(f"\nAccuracy:           {accuracy*100:.2f}%")
    print(f"Precision:          {precision_macro:.4f}")
    print(f"Recall:             {recall_macro:.4f}")
    print(f"F1-Score:           {f1_macro:.4f}")
    print(f"ROC-AUC:            {macro_roc_auc:.4f}\n")
    
    # VISUALIZATIONS
    print(f"{'-'*70}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'-'*70}\n")
    
    # ============================================================================
    # VISUALIZATION 1: COMPREHENSIVE 9-PANEL DASHBOARD
    # ============================================================================
    print("1. Creating comprehensive 9-panel evaluation dashboard...")
    
    fig = plt.figure(figsize=(18, 14))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(3, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gesture_names, yticklabels=gesture_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Per-Class Accuracy
    ax2 = plt.subplot(3, 3, 2)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = ['green' if acc == 1.0 else 'orange' if acc > 0.9 else 'red' for acc in per_class_acc]
    ax2.barh(gesture_names, per_class_acc, color=colors, edgecolor='black')
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Per-Class Accuracy', fontsize=13, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    for i, v in enumerate(per_class_acc):
        ax2.text(v + 0.02, i, f'{v:.2%}', va='center', fontsize=9)
    
    # 3. ROC-AUC Scores (Bar Chart)
    ax3 = plt.subplot(3, 3, 3)
    roc_values = [roc_auc_dict[i] for i in range(num_classes)]
    colors_roc = ['green' if score == 1.0 else 'orange' if score > 0.95 else 'red' for score in roc_values]
    ax3.barh(gesture_names, roc_values, color=colors_roc, edgecolor='black')
    ax3.set_xlabel('ROC-AUC Score')
    ax3.set_title('ROC-AUC Scores', fontsize=13, fontweight='bold')
    ax3.set_xlim([0.9, 1.01])
    for i, v in enumerate(roc_values):
        ax3.text(v - 0.015, i, f'{v:.4f}', va='center', ha='right', 
                color='white', fontweight='bold', fontsize=8)
    
    # 4. ROC Curves (One-vs-Rest)
    ax4 = plt.subplot(3, 3, 4)
    colors_list = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        if i in fpr_dict:
            ax4.plot(fpr_dict[i], tpr_dict[i], color=colors_list[i], 
                    label=f'{gesture_names[i]} (AUC = {roc_auc_dict[i]:.3f})', linewidth=2)
    
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves (One-vs-Rest)', fontsize=13, fontweight='bold')
    ax4.legend(loc='lower right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Confidence Distribution
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
            label=f'Correct (μ={avg_conf_correct:.3f})', color='green', edgecolor='black')
    ax5.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
            label=f'Incorrect (μ={avg_conf_incorrect:.3f})', color='red', edgecolor='black')
    ax5.axvline(0.70, color='black', linestyle='--', linewidth=2, label='Threshold (70%)')
    ax5.set_xlabel('Confidence Score')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Prediction Confidence Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Precision vs Recall Scatter
    ax6 = plt.subplot(3, 3, 6)
    precisions = [per_class_metrics[g]['precision'] for g in gesture_names]
    recalls = [per_class_metrics[g]['recall'] for g in gesture_names]
    
    ax6.scatter(recalls, precisions, s=300, alpha=0.7, c=range(num_classes), 
               cmap='tab10', edgecolors='black', linewidth=2)
    for i, gesture in enumerate(gesture_names):
        ax6.annotate(gesture, (recalls[i], precisions[i]), fontsize=8, ha='center', fontweight='bold')
    ax6.set_xlabel('Recall')
    ax6.set_ylabel('Precision')
    ax6.set_title('Precision vs Recall', fontsize=13, fontweight='bold')
    ax6.set_xlim([0.85, 1.05])
    ax6.set_ylim([0.85, 1.05])
    ax6.grid(True, alpha=0.3)
    ax6.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    
    # 7. Precision-Recall Curves
    ax7 = plt.subplot(3, 3, 7)
    for i in range(num_classes):
        precision_curve, recall_curve, _ = precision_recall_curve(
            y_test_bin[:, i], y_pred_proba[:, i]
        )
        ap = auc(recall_curve, precision_curve)
        ax7.plot(recall_curve, precision_curve, color=colors_list[i], linewidth=2,
                label=f'{gesture_names[i]} (AP = {ap:.3f})')
    
    ax7.set_xlabel('Recall')
    ax7.set_ylabel('Precision')
    ax7.set_title('Precision-Recall Curves', fontsize=13, fontweight='bold')
    ax7.legend(loc='best', fontsize=8)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim([-0.05, 1.05])
    ax7.set_ylim([-0.05, 1.05])
    
    # 8. F1-Score Comparison
    ax8 = plt.subplot(3, 3, 8)
    f1_scores = [per_class_metrics[g]['f1'] for g in gesture_names]
    colors_f1 = ['green' if f1 > 0.95 else 'orange' if f1 > 0.85 else 'red' for f1 in f1_scores]
    ax8.barh(gesture_names, f1_scores, color=colors_f1, edgecolor='black')
    ax8.set_xlabel('F1-Score')
    ax8.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax8.set_xlim([0.75, 1.05])
    for i, v in enumerate(f1_scores):
        ax8.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    # 9. Summary Statistics
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""MODEL EVALUATION SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 OVERALL METRICS:
  • Accuracy:        {accuracy:.4f} ({accuracy*100:.2f}%)
  • Precision:       {precision_macro:.4f}
  • Recall:          {recall_macro:.4f}
  • F1-Score:        {f1_macro:.4f}
  • ROC-AUC:         {macro_roc_auc:.4f}

 TEST DATA:
  • Total Samples:   {len(y_test_labels)}
  • Correct:         {(y_pred == y_test_labels).sum()}
  • Incorrect:       {num_misclassified}

 MODEL CONFIG:
  • Classes:         {num_classes} gestures
  • Conf. Gap:       {(avg_conf_correct - avg_conf_incorrect):.4f}
  • Best:            {max(per_class_metrics.items(), key=lambda x: x[1]['f1'])[0]}
  • Worst:           {min(per_class_metrics.items(), key=lambda x: x[1]['f1'])[0]}
"""
    ax9.text(0.05, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='lightyellow', alpha=0.8, pad=1.5), fontweight='bold')
    
    plt.suptitle('Comprehensive Model Evaluation Report', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: evaluation_metrics.png (9-panel dashboard)\n")
    
    # ============================================================================
    # VISUALIZATION 2: STANDALONE ROC-AUC CURVES (FOR PAPER)
    # ============================================================================
    print("2. Creating standalone ROC-AUC curves visualization...")
    
    fig_roc = plt.figure(figsize=(12, 8))
    
    # Define colors for ROC curves
    colors_roc_plot = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Plot ROC curves
    for i in range(num_classes):
        if i in fpr_dict:
            plt.plot(fpr_dict[i], tpr_dict[i], 
                    color=colors_roc_plot[i], lw=2.8, 
                    label=f'{gesture_names[i]} (AUC = {roc_auc_dict[i]:.4f})')
    
    # Plot random classifier line
    plt.plot([0, 1], [0, 1], 'k--', lw=2.5, label='Random Classifier (AUC = 0.5000)', alpha=0.7)
    
    # Formatting
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate (FPR)', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (TPR)', fontsize=13, fontweight='bold')
    plt.title('ROC-AUC Curves (One-vs-Rest Classification)\nMacro Average AUC = {:.4f}'.format(macro_roc_auc), 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
    
    plt.tight_layout()
    plt.savefig('roc_auc_curves.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: roc_auc_curves.png (ROC curves - FOR YOUR PAPER!)\n")
    
    # ============================================================================
    # VISUALIZATION 3: HIGH-QUALITY CONFUSION MATRIX
    # ============================================================================
    print("3. Creating high-quality confusion matrix visualization...")
    
    fig_cm = plt.figure(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=gesture_names, yticklabels=gesture_names,
                annot_kws={'size': 14, 'fontweight': 'bold'}, cbar=True,
                linewidths=1.5, linecolor='white')
    plt.title('Confusion Matrix - Gesture Classification Results', fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(rotation=0, fontsize=11)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("   ✓ Saved: confusion_matrix.png (High-quality confusion matrix)\n")
    
    # ============================================================================
    # EVALUATION COMPLETE
    # ============================================================================
    print(f"{'-'*70}")
    print("✓ EVALUATION COMPLETE!")
    print(f"{'-'*70}\n")
    print("Generated Files:")
    print("  ✓ evaluation_metrics.png    (Comprehensive 9-panel dashboard)")
    print("  ✓ roc_auc_curves.png        (ROC-AUC curves - USE IN PAPER!)")
    print("  ✓ confusion_matrix.png      (High-quality confusion matrix)\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'roc_auc': macro_roc_auc,
        'per_class_metrics': per_class_metrics,
        'roc_scores': roc_scores,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_test_labels': y_test_labels,
        'confidence_scores': confidence_scores
    }

if __name__ == "__main__":
    results = evaluate_model()
    print(f"{'-'*70}")
    print("All metrics and visualizations generated successfully!")
    print(f"→ Use 'roc_auc_curves.png' in your paper!")
    print(f"{'-'*70}\n")
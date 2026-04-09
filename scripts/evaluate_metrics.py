"""
Comprehensive Model Evaluation Script
Generates: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, Visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
import pickle
import seaborn as sns

def evaluate_model():
    """Calculate all important metrics and visualizations"""
    
    print(f"\n{'-'*70}")
    print("COMPREHENSIVE MODEL EVALUATION")
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
    for i, gesture in enumerate(gesture_names):
        try:
            roc_auc = roc_auc_score(y_test_bin[:, i], y_pred_proba[:, i])
            roc_scores[gesture] = roc_auc
            print(f"{gesture:<15} {roc_auc:>20.4f}")
        except:
            print(f"{gesture:<15} {'N/A':>20}")
    
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
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Confusion Matrix
    ax1 = plt.subplot(2, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=gesture_names, yticklabels=gesture_names,
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Per-Class Accuracy
    ax2 = plt.subplot(2, 3, 2)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = ['green' if acc == 1.0 else 'orange' if acc > 0.9 else 'red' for acc in per_class_acc]
    ax2.barh(gesture_names, per_class_acc, color=colors)
    ax2.set_xlabel('Accuracy')
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1.05])
    for i, v in enumerate(per_class_acc):
        ax2.text(v + 0.02, i, f'{v:.2%}', va='center', fontsize=10)
    
    # 3. ROC-AUC Scores
    ax3 = plt.subplot(2, 3, 3)
    roc_values = list(roc_scores.values())
    colors_roc = ['green' if score == 1.0 else 'orange' if score > 0.95 else 'red' for score in roc_values]
    ax3.barh(gesture_names, roc_values, color=colors_roc)
    ax3.set_xlabel('ROC-AUC Score')
    ax3.set_title('ROC-AUC Scores', fontsize=14, fontweight='bold')
    ax3.set_xlim([0.9, 1.01])
    for i, v in enumerate(roc_values):
        ax3.text(v - 0.015, i, f'{v:.4f}', va='center', ha='right', 
                color='white', fontweight='bold', fontsize=9)
    
    # 4. Confidence Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(confidence_scores[correct_predictions], bins=20, alpha=0.7, 
            label='Correct', color='green', edgecolor='black')
    ax4.hist(confidence_scores[~correct_predictions], bins=20, alpha=0.7, 
            label='Incorrect', color='red', edgecolor='black')
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Precision vs Recall
    ax5 = plt.subplot(2, 3, 5)
    precisions = [per_class_metrics[g]['precision'] for g in gesture_names]
    recalls = [per_class_metrics[g]['recall'] for g in gesture_names]
    
    ax5.scatter(recalls, precisions, s=200, alpha=0.6, c=range(num_classes), cmap='tab10')
    for i, gesture in enumerate(gesture_names):
        ax5.annotate(gesture, (recalls[i], precisions[i]), fontsize=9, ha='center')
    ax5.set_xlabel('Recall')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision vs Recall', fontsize=14, fontweight='bold')
    ax5.set_xlim([0.85, 1.05])
    ax5.set_ylim([0.85, 1.05])
    ax5.grid(True, alpha=0.3)
    ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    
    # 6. Summary Statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    summary_text = f"""
    MODEL EVALUATION SUMMARY
    {'='*45}
    
    Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)
    Precision:          {precision_macro:.4f}
    Recall:             {recall_macro:.4f}
    F1-Score:           {f1_macro:.4f}
    ROC-AUC:            {macro_roc_auc:.4f}
    
    Test Samples:       {len(y_test_labels)}
    Correct:            {(y_pred == y_test_labels).sum()}
    Incorrect:          {num_misclassified}
    
    Classes:            {num_classes} gestures
    """
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='lightblue', alpha=0.7, pad=1))
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization: evaluation_metrics.png\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'roc_auc': macro_roc_auc,
        'per_class_metrics': per_class_metrics
    }

if __name__ == "__main__":
    results = evaluate_model()
    print(f"{'-'*70}")
    print("EVALUATION COMPLETE!")
    print(f"{'-'*70}\n")
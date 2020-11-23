from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
import numpy as np
from matplotlib import pyplot as plt

def calc_metrics(pred_vec, label_vec):
    """
    Calculate AUC, mIoU and accuracy
    """

    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((label_vec), pred_vec)
    AUC_ROC = roc_auc_score(label_vec, pred_vec)
    print(f'AUC: {AUC_ROC:.6f}')
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.zeros_like(pred_vec)
    y_pred[pred_vec > threshold_confusion] = 1
    confusion = confusion_matrix(label_vec, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print(f'Accuracy: {accuracy:.6f}')
    mIoU = confusion[1, 1] / (confusion[1, 1] + confusion[0, 1] + confusion[1, 0])
    print(f'mIoU: {mIoU:.6f}')
    dice_loss = 2 * confusion[1, 1] / (2 * confusion[1, 1] + confusion[0, 1] + confusion[1, 0])
    print(f'Dice loss: {dice_loss:.6f}')

    plt.plot(fpr, tpr, '-')
    plt.title('ROC curve', fontsize=14)
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.savefig('ROC.png')
    plt.show()
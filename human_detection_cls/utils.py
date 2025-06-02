import seaborn as sns
import os.path as osp
import matplotlib.pyplot as plt

def plot_label_counts(df, artifact_dir, flag):
    sns.countplot(x='labels', data=df)
    plt.title("Count of each label")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(osp.join(artifact_dir, f"{flag}_label_counts.png"))
    plt.close()
    
def plot_metrics(train_loss, val_loss, train_acc, val_acc, epoch, dir_path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].plot(train_loss, label="Train Loss", color="r")
    axes[0].plot(val_loss, label="Validation Loss", color="b")
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(train_acc, label="Train Accuracy", color="g")
    axes[1].plot(val_acc, label="Validation Accuracy", color="orange")
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(osp.join(dir_path, f"metrics_plot_epoch_{epoch}.png"))
    plt.close()

def plot_class_wise_ap_auroc(ap_history, auroc_history, classwise_precision, classwise_recall, classwise_f1_score, epoch, dir_path):

    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    axes.plot(ap_history, label="mAP", color="r")
    axes.plot(auroc_history, label="auroc", color="g")
    axes.plot(classwise_precision, label="precision", color="b")
    axes.plot(classwise_recall, label="recall", color="y")
    axes.plot(classwise_f1_score, label="f1_score", color="m")
    axes.set_title("mAP, Auroc, Precision, Recall, F1 score over Epochs")
    axes.set_xlabel("Epochs")
    axes.set_ylabel("mAP & Auroc")
    axes.legend()
    axes.grid(True)

    plt.tight_layout()
    plt.savefig(osp.join(dir_path, f"auroc_mAP_precision_recall_f1score_{epoch}.png"))
    plt.close()

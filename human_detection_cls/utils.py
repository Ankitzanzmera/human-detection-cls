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

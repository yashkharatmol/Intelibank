import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, labels, filename="confusion_matrix.png"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(filename)

def extract_keywords_and_generate_label(text):
    import yake
    kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=3)
    keywords = kw_extractor.extract_keywords(text)
    return ", ".join([kw[0] for kw in keywords])

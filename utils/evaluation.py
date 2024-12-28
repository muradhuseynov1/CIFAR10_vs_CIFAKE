import torch
from sklearn.metrics import classification_report, confusion_matrix
import os
import json

#this is evaluation function that we used in most of our trainings

def evaluate_model(model, dataloader, device, phase="validation", metrics_file_path="metrics/metrics.txt"):
    # we always saved metrics.txt in metrics folder of each branch
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    classification_rep = classification_report(all_labels, all_preds, output_dict=True)
    confusion_mat = confusion_matrix(all_labels, all_preds)

    # create metrics if it doesn't exsist yet
    os.makedirs(os.path.dirname(metrics_file_path), exist_ok=True)

    # write everything down
    with open(metrics_file_path, "a") as f:
        f.write(f"Phase: {phase}\n")
        f.write("Classification Report:\n")
        for class_label, metrics in classification_rep.items():
            if isinstance(metrics, dict):  # Avoid overall averages like accuracy
                f.write(f"Class {class_label}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_mat))
        f.write("\n" + "="*50 + "\n")

    print("Classification Report:")
    print(classification_report(all_labels, all_preds))
    print("Confusion Matrix:")
    print(confusion_mat)
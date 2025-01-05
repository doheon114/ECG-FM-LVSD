import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from fairseq_signals.models import build_model_from_checkpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, ConfusionMatrixDisplay
import json



# GPU and model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

with open("path/to/data_(4,5000).pkl", "rb") as f:
    data = pickle.load(f)

X_int = torch.tensor(data["int test"]["x"], dtype=torch.float32)
y_int = torch.tensor(data["int test"]["y"], dtype=torch.float32)
X_ext_one = torch.tensor(data["ext test 1"]["x"], dtype=torch.float32)
y_ext_one = torch.tensor(data["ext test 1"]["y"], dtype=torch.float32)
X_ext_two = torch.tensor(data["ext test 2"]["x"], dtype=torch.float32)
y_ext_two = torch.tensor(data["ext test 2"]["y"], dtype=torch.float32)

th = 0.4
y_int_binary = (y_int < th).float()
y_ext_one_binary = (y_ext_one < th).float()
y_ext_two_binary = (y_ext_two < th).float()


# Model initialization
model_pretrained = build_model_from_checkpoint(
    checkpoint_path=(os.path.join('checkpoints/pretrained.pt'))
).to(device)  # Move model to GPU

class FineTunedWav2Vec2Model(nn.Module):
    def __init__(self, pretrained_model):
        super(FineTunedWav2Vec2Model, self).__init__()
        self.pretrained_model = pretrained_model
        self.conv1d_first = nn.Conv1d(4, 4, kernel_size=1)
        self.conv1d = nn.Conv1d(4, 12, kernel_size=1)
        self.pretrained_model.proj = nn.Linear(self.pretrained_model.proj.in_features, 1)

    def forward(self, source):
        source = self.conv1d_first(source)
        source = self.conv1d(source)
        outputs = self.pretrained_model(source=source)
        outputs = outputs['out']

        return outputs

model_with_classification_head = FineTunedWav2Vec2Model(pretrained_model=model_pretrained)


if torch.cuda.device_count() > 1:
    model_with_classification_head = nn.DataParallel(model_with_classification_head)

# Loads the fine-tuned model weights from train.py.
model_with_classification_head.load_state_dict(torch.load('checkpoints/ECG-FM(SS-FF)_4channels.pt'))

model_with_classification_head.to(device)

# Evaluate with validation data
model_with_classification_head.eval()

def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path):
    """
    Function to visualize and save Confusion Matrix
    Args:
        y_true (np.array): Actual labels
        y_pred (np.array): Predicted labels
        class_names (list): List of class names
        title (str): Title of the plot
        save_path (str): Save path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Reverse the positions of positive and negative for Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=class_names[::-1],  # Reverse class name order
        cmap=plt.cm.Blues,
    )
    
    plt.title(title)
    plt.gca().invert_yaxis()  # Invert y-axis
    plt.gca().invert_xaxis()  # Invert x-axis
    plt.savefig(save_path)
    plt.close()


# Performance evaluation and confusion matrix visualization
def evaluate_model(model, X, y_binary, dataset_name, device):
    X = X.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(source=X)
        logits = outputs.squeeze()
        predictions = (torch.sigmoid(logits) >= 0.5).float()


        tp, fn, fp, tn = confusion_matrix(y_binary.cpu(), predictions.cpu()).ravel()

       
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tn / (tn+ fn) if (tn + fn) > 0 else 0
        NPV = tp / (tp + fp) if (tp +fp) > 0 else 0
        specificity = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        auc_roc = roc_auc_score(y_binary.cpu(), torch.sigmoid(logits).cpu())
        auprc = average_precision_score(y_binary.cpu(), torch.sigmoid(logits).cpu())

        print(f"Accuracy ({dataset_name}): {accuracy:.4f}")
        print(f"Sensitivity ({dataset_name}): {sensitivity:.4f}")
        print(f"Precision ({dataset_name}): {precision:.4f}")
        print(f"NPV ({dataset_name}): {NPV:.4f}")
        print(f"Specificity({dataset_name}): {specificity:.4f}")
        print(f"F1 Score ({dataset_name}): {f1:.4f}")
        print(f"AUROC ({dataset_name}): {auc_roc:.4f}")
        print(f"AUPRC ({dataset_name}): {auprc:.4f}")

        # Visualize and save confusion matrix
        plot_confusion_matrix(
            y_true=y_binary.cpu().numpy(),
            y_pred=predictions.cpu().numpy(),
            class_names=['LVSD', 'NoLVSD'],
            title=f"Confusion Matrix ({dataset_name})",
            save_path=f"path/to/save/confusion_matrix_{dataset_name.lower()}.png"
        )

        # Save performance metrics in a dictionary
        return {
            "Dataset": dataset_name,
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Precision": precision,
            "NPV": NPV,
            "Specificity": specificity,
            "F1 Score": f1,
            "AUROC": auc_roc,
            "AUPRC": auprc
        }


# Save evaluation results for each test set
def save_metrics_to_file(model, device):
    results = []
    # Perform evaluation for each test set and save results
    results.append(evaluate_model(model, X_int, torch.tensor(y_int_binary, dtype=torch.float32), "Internal Validation Set", device))
    results.append(evaluate_model(model, X_ext_one, torch.tensor(y_ext_one_binary, dtype=torch.float32), "External Validation Set 1", device))
    results.append(evaluate_model(model, X_ext_two, torch.tensor(y_ext_two_binary, dtype=torch.float32), "External Validation Set 2", device))

    # Save results to a JSON file
    with open("path/to/save/test_set_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Performance metrics have been saved to test_set_metrics.json file.")

# Example function call
save_metrics_to_file(model_with_classification_head, device)

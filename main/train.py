import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from fairseq_signals.models import build_model_from_checkpoint
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pretrained model
model_pretrained = build_model_from_checkpoint(
    checkpoint_path=(os.path.join('checkpoints/pretrained.pt'))
).to(device)  # Move model to GPU

# Load data
with open("path/to/your/own/data.pkl", "rb") as f:
    data = pickle.load(f)
X_train = torch.tensor(data["train"]["x"], dtype=torch.float32)
y_train = torch.tensor(data["train"]["y"], dtype=torch.float32)

# Binary target setting (0 or 1)
th = 0.4  # Threshold for LVSD classification based on LVEF value
y_train_binary = (y_train < th).float()

X_train, X_val, y_train_binary, y_val_binary = train_test_split(X_train, y_train_binary, stratify=y_train_binary, test_size=0.2)

# Create TensorDataset
train_dataset = TensorDataset(X_train, torch.tensor(y_train_binary, dtype=torch.float32))
val_dataset = TensorDataset(X_val, torch.tensor(y_val_binary, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=False)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)


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

# Count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

trainable_params = count_trainable_parameters(model_with_classification_head)
print(f"Trainable parameters: {trainable_params}")

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    model_with_classification_head = nn.DataParallel(model_with_classification_head)

model_with_classification_head.to(device)
optimizer = optim.Adam(model_with_classification_head.parameters(), lr=5e-7, betas=(0.9, 0.98))
criterion = F.binary_cross_entropy_with_logits

train_loss, val_loss, train_accuracy_list, val_accuracy_list = [], [], [], []
num_epochs = 60
best_val_loss = float('inf')  # Initialize best validation loss

# Training loop
for epoch in range(num_epochs):
    model_with_classification_head.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch in tqdm(train_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        positive_count = labels.sum().item()
        negative_count = len(labels) - positive_count
        pos_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32).to(device) if positive_count > 0 else torch.tensor([1.0], dtype=torch.float32).to(device)

        optimizer.zero_grad()
        outputs = model_with_classification_head(source=inputs)
        logits = outputs.squeeze()
        loss = criterion(logits, labels, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate accuracy
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct_predictions += (preds == labels).sum().item()
        total_samples += labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = correct_predictions / total_samples
    train_loss.append(epoch_loss)
    train_accuracy_list.append(train_accuracy)

    # Validation loop
    model_with_classification_head.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for val_batch in val_loader:
            val_inputs, val_labels = val_batch
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            val_outputs = model_with_classification_head(source=val_inputs)
            val_logits = val_outputs.squeeze()
            val_predictions = (torch.sigmoid(val_logits) >= 0.5).float()
            val_correct += (val_predictions == val_labels).sum().item()
            val_total += val_labels.size(0)
            val_loss_value = criterion(val_logits, val_labels)
            val_running_loss += val_loss_value.item()

    epoch_val_loss = val_running_loss / len(val_loader)
    val_loss.append(epoch_val_loss)
    val_accuracy = val_correct / val_total
    val_accuracy_list.append(val_accuracy)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# Plot training and validation loss
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_loss, label='Train Loss')
plt.plot(np.arange(1, num_epochs + 1), val_loss, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.plot(np.arange(1, num_epochs + 1), train_accuracy_list, label='Training Accuracy', color='blue')
plt.plot(np.arange(1, num_epochs + 1), val_accuracy_list, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Save model weights
def save_final_model(model, save_path):
    """
    Save the final model weights to the specified path.
    Args:
        model (nn.Module): Trained model
        save_path (str): Path to save the weights
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Final model weights saved at '{save_path}'.")

save_final_model(model_with_classification_head, "path/to/save/weights.pth")

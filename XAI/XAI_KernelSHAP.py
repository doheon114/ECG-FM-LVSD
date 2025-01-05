import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
from fairseq_signals.models import build_model_from_checkpoint
import os
from captum.attr import KernelShap

# ------------------------------
# 1. Setup and Data Loading
# ------------------------------

# Set GPU and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained model
model_pretrained = build_model_from_checkpoint(
    checkpoint_path=(os.path.join('checkpoints/pretrained.pt'))
).to(device)  # Move model to GPU

# Load data
with open("path/to/data_(4,5000).pkl", "rb") as f:
    data = pickle.load(f)

# Extract internal and external test datasets
X_int = torch.tensor(data["int test"]["x"], dtype=torch.float32)
y_int = torch.tensor(data["int test"]["y"], dtype=torch.float32)
X_ext_one = torch.tensor(data["ext test 1"]["x"], dtype=torch.float32)
y_ext_one = torch.tensor(data["ext test 1"]["y"], dtype=torch.float32)
X_ext_two = torch.tensor(data["ext test 2"]["x"], dtype=torch.float32)
y_ext_two = torch.tensor(data["ext test 2"]["y"], dtype=torch.float32)

# ------------------------------
# 2. Model Loading
# ------------------------------

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

# Attach classification head to pretrained model
model = FineTunedWav2Vec2Model(pretrained_model=model_pretrained)

# Use multiple GPUs if available
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

model.to(device)

# Load fine-tuned model weights
model.load_state_dict(torch.load('checkpoints/ECG-FM(SS-FF)_4channels.pt'))

# Set the model to evaluation mode
model.eval()

# ------------------------------
# 3. SHAP Value Calculation
# ------------------------------

X_10 = X_int[:194]

# Move data to GPU
X_10.to(device)

# Verify model compatibility with inputs
inputs = X_10
output = model(inputs)

# Set baseline values for SHAP
baselines = torch.zeros_like(inputs)

# Initialize Kernel SHAP
dl = KernelShap(model)
shap_values = dl.attribute(inputs, baselines)

# ------------------------------
# 4. Visualization
# ------------------------------

sample_number = range(0, 10)  # Index of the sample to visualize
for i in sample_number:
    sample_idx = i
    target_class_idx = 0  # Target class index

    # Extract ECG data for the selected sample
    ecg_data = X_10[sample_idx]
    num_leads, signal_length = ecg_data.shape

    # Extract SHAP values for the target class
    if isinstance(shap_values, list):
        target_shap_values = shap_values[target_class_idx][sample_idx]
    else:
        target_shap_values = shap_values[sample_idx, :, :]

    # Convert ECG data to PyTorch tensor
    ecg_tensor = torch.tensor(np.expand_dims(ecg_data, axis=0), dtype=torch.float32)

    # Model prediction
    predicted_class = np.argmax(model(ecg_tensor).detach().cpu().numpy(), axis=1)[0]

    # Identify top N SHAP values
    N = 200  # Number of top SHAP values to plot
    flat_indices = np.argsort(np.abs(target_shap_values.detach().cpu().numpy()).ravel())[-N:]
    time_indices, lead_indices = np.unravel_index(flat_indices, (5000, 4))

    # Create time array
    sampling_rate = 500  # Hz
    time = np.linspace(0, signal_length / sampling_rate, signal_length)

    # Initialize plot
    fig, axes = plt.subplots(num_leads, 1, figsize=(15, 10), sharex=True)

    # Determine global SHAP scaling factor for consistent visualization
    shap_scale_factor = (torch.max(ecg_data) - torch.min(ecg_data)).item() * 0.1

    # Plot each lead with SHAP overlay
    for lead_idx in range(num_leads):
        lead_data = ecg_data[lead_idx, :]
        axes[lead_idx].plot(time, lead_data, lw=1, color='black')
        axes[lead_idx].set_ylim([-2, 2])

        # Find top SHAP indices for this lead
        current_lead_mask = lead_indices == lead_idx
        current_time_indices = time_indices[current_lead_mask]
        current_shap_values = target_shap_values[lead_idx, current_time_indices]

        # Scale SHAP values
        scaled_shap_values = current_shap_values * shap_scale_factor

        # Overlay SHAP values
        shap_overlay = lead_data[current_time_indices]
        time_for_scatter = time[current_time_indices]
        shap_overlay = shap_overlay.detach().cpu().numpy()

        # Scatter plot of SHAP values
        axes[lead_idx].scatter(time_for_scatter, shap_overlay, color='orange', s=20, alpha=0.9)

    # Final plot adjustments
    axes[-1].set_xlabel('Time (seconds)')
    plt.suptitle(f'ECG Signals with Top 200 SHAP Values', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save and display plot
    plt.savefig(f"path/to/save/results/{i+1}_top200features.png")
    plt.show()

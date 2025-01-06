# ECG-FM-LVSD

![License](https://img.shields.io/github/license/doheon114/ECG-FM-LVSD)
![Stars](https://img.shields.io/github/stars/doheon114/ECG-FM-LVSD)
![Forks](https://img.shields.io/github/forks/doheon114/ECG-FM-LVSD)

## Overview

ECG-FM-LVSD is a project focused on predicting left ventricular systolic dysfunction (LVSD) using ECG data and explainable artificial intelligence (XAI) techniques. This repository includes tools for model training, evaluation, and interpretability to enhance understanding of how predictions are made.

---

## Features

- **ECG-Based Prediction**: Utilize ECG data for accurate LVSD prediction.
- **Explainable AI (XAI)**: Incorporate interpretability methods to explain model predictions and uncover relevant features.
- **Customizable Models**: Easily modify and fine-tune ECG-FM for LVSD risk prediction.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/doheon114/ECG-FM-LVSD.git
cd ECG-FM-LVSD
```

---

## Directory Structure

```
ECG-FM-LVSD/
├── XAI/                    # code for eXplainable AI using KernelSHAP 
├── checkpoints/            # pretrained model & finetuned best model weights
├── main/                   # code containing the key aspects of the experiment
└── results/                # Evaluation results and explanations
```

---

## Usage

### 1. Train the Model

Finetune ECG-FM model for LVSD prediction:
```bash
python main/train.py 
```

### 2. Evaluate the Model

Evaluate the model's performance on a validation dataset:
```bash
python main/evaluate.py 
```

### 3. Explain Predictions

Generate explanations for model predictions using XAI techniques:
```bash
python XAI/explain.py 
```

---


## Requirements

First, Clone the [fairseq_signals](https://github.com/Jwoo5/fairseq-signals.git) repository and follow the instructions in the requirements and installation section of the main README file.

The project requires the following additional Python packages:

- scikit-learn
- matplotlib    
- captum (for XAI)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Contact

For questions or feedback, please contact:
- **Author**: [doheon114](https://github.com/doheon114)
- **Email**: doheon135790@gmail.com

---

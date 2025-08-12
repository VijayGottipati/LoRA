# LoRA Fine-tuning for Text Classification

A comprehensive implementation of Low-Rank Adaptation (LoRA) fine-tuning for text classification using RoBERTa on the AG News dataset. This project demonstrates efficient parameter-efficient fine-tuning with hyperparameter optimization using Optuna.

## 🚀 Overview

This project implements LoRA (Low-Rank Adaptation) fine-tuning on a pre-trained RoBERTa model for news article classification. LoRA allows efficient fine-tuning by introducing trainable low-rank matrices while keeping the original model parameters frozen, significantly reducing the number of trainable parameters and computational requirements.

## 📊 Dataset

- **Dataset**: AG News Classification Dataset
- **Task**: 4-class news categorization
- **Classes**: World, Sports, Business, Science/Technology
- **Training samples**: ~108,000 (after 90/10 split)
- **Validation samples**: ~12,000
- **Test samples**: 7,600

## 🏗️ Model Architecture

- **Base Model**: RoBERTa-base (125M parameters)
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Target Modules**: Query and Value projection layers
- **Task**: Sequence Classification

## ⚙️ Key Features

- **Parameter-Efficient Fine-tuning**: Uses LoRA to reduce trainable parameters by ~99%
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **GPU Acceleration**: CUDA support with mixed precision training
- **Comprehensive Evaluation**: Accuracy metrics and detailed logging
- **Reproducible Results**: Fixed random seeds and deterministic training

## 🔧 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory

### Dependencies

```bash
pip install transformers datasets evaluate accelerate peft bitsandbytes trl optuna
pip install nvidia-ml-py3
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📈 Hyperparameter Optimization

The project uses Optuna for automated hyperparameter tuning with the following search space:

| Parameter | Search Space | Best Value |
|-----------|--------------|------------|
| LoRA Rank (r) | [4, 8, 16] | 8 |
| LoRA Alpha | [8, 16, 32] | 16 |
| LoRA Dropout | [0.05, 0.2] | ~0.15 |
| Learning Rate | [1e-5, 2e-5, 3e-5] | 2e-5 |
| Epochs | [2, 3] | 3 |

## 🎯 Results

- **Best Validation Accuracy**: 91.67%
- **Training Time**: ~2.5 hours per trial (3 epochs)
- **Trainable Parameters**: ~0.3M (vs 125M for full fine-tuning)
- **Memory Efficiency**: ~70% reduction in GPU memory usage

## 🚀 Usage

### Quick Start

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/LoRA.git
cd LoRA
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the training**:
```bash
jupyter notebook deepl2.ipynb
```

### Training Process

The notebook includes:

1. **Environment Setup**: Installing required packages
2. **Data Loading**: AG News dataset preparation and tokenization
3. **Model Configuration**: LoRA setup with RoBERTa
4. **Hyperparameter Tuning**: Optuna optimization
5. **Final Training**: Training with best parameters
6. **Evaluation**: Model performance assessment
7. **Inference**: Predictions on test data

### Key Training Parameters

```python
# LoRA Configuration
peft_config = LoraConfig(
    r=8,                    # Low-rank dimension
    lora_alpha=16,          # LoRA scaling parameter
    lora_dropout=0.15,      # Dropout rate
    bias='none',            # Bias handling
    target_modules=['query', 'value'],  # Target attention layers
    task_type="SEQ_CLS"     # Sequence classification
)

# Training Arguments
training_args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    warmup_steps=500,
    gradient_accumulation_steps=2,
    lr_scheduler_type="cosine"
)
```

## 📁 Project Structure

```
LoRA/
├── README.md                 # Project documentation
├── deepl2.ipynb            # Main training notebook
├── requirements.txt         # Python dependencies
├── experiment_log.csv       # Hyperparameter tuning results
├── submission.csv          # Test predictions
└── results/                # Training outputs and checkpoints
```

## 🔬 Technical Details

### LoRA Implementation

LoRA introduces trainable low-rank matrices A and B such that:
```
W = W₀ + BA
```

Where:
- W₀: Frozen pre-trained weights
- B: Trainable matrix (d × r)
- A: Trainable matrix (r × k)
- r: Low-rank dimension (much smaller than d, k)

### Memory and Computational Benefits

- **Parameter Reduction**: 99.7% fewer trainable parameters
- **Memory Efficiency**: ~70% reduction in GPU memory usage
- **Training Speed**: 2-3x faster training compared to full fine-tuning
- **Storage**: Minimal storage for adapter weights (~2MB vs 500MB)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| Validation Accuracy | 91.67% |
| Training Loss | 0.352 |
| Validation Loss | 0.243 |
| F1-Score | ~0.916 |
| Training Time | 2.5 hours |

## 🛠️ Advanced Configuration

### Custom Hyperparameter Tuning

Modify the `objective` function in the notebook to explore different hyperparameter ranges:

```python
def objective(trial):
    lora_r = trial.suggest_categorical('lora_r', [4, 8, 16, 32])
    lora_alpha = trial.suggest_categorical('lora_alpha', [8, 16, 32, 64])
    lora_dropout = trial.suggest_float('lora_dropout', 0.05, 0.3)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-6, 1e-4)
    # ... rest of the function
```

### Multi-GPU Training

For multi-GPU setups, modify the training arguments:

```python
training_args = TrainingArguments(
    # ... other arguments
    dataloader_num_workers=4,
    ddp_find_unused_parameters=False,
    # Enable for multi-GPU
)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the excellent library
- [Microsoft LoRA](https://github.com/microsoft/LoRA) for the original LoRA implementation
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [AG News Dataset](https://huggingface.co/datasets/ag_news) for the classification task

## 📚 References

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv preprint arXiv:2106.09685.
2. Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv preprint arXiv:1907.11692.
3. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NIPS.

## 📞 Contact

- **Author**: [Your Name]
- **Email**: [your.email@example.com]
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

⭐ If you found this project helpful, please give it a star!
# EEG Classification Framework

A comprehensive framework for EEG (Electroencephalogram) signal classification with deep learning models, specifically designed for BCI (Brain-Computer Interface) applications.

## Features

- Support for multiple EEG datasets, including BCICIV2A
- Implementation of state-of-the-art models such as EEG-Conformer
- Cross-session and cross-subject evaluation paradigms
- Data augmentation techniques for improved generalization
- Experiment tracking with Weights & Biases
- Comprehensive visualization tools (t-SNE, confusion matrices)
- Model registry for easy model management and extension

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/eeg-classification-framework.git
   cd eeg-classification-framework
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Weights & Biases:
   ```bash
   pip install wandb
   wandb login
   ```

## Project Structure

```
lesson1_complex/
├── codes/
│   ├── dataset/
│   │   ├── augmentation.py       # Data augmentation techniques
│   │   └── dataset_class.py      # Dataset handling classes
│   ├── draw/
│   │   ├── Confusion_matrix.py   # Confusion matrix visualization
│   │   └── TSNE.py               # t-SNE visualization
│   ├── models/
│   │   ├── EEG_conformer.py      # EEG-Conformer model
│   │   ├── EEG_conformer_CTN.py  # EEG-Conformer with CTN
│   │   └── ModelResgistry.py     # Model registration system
│   ├── paradigm/
│   │   └── dataset_2a_paradigm.py # Training paradigms
│   ├── main.py                   # Entry point
│   ├── preprocessing.py          # Data preprocessing
│   └── trainer.py                # Training pipeline
├── configs/                      # Model configuration files
├── logs/                         # Training logs
├── models/                       # Saved models
└── results/                      # Evaluation results
```

## Usage

### Basic Training

```bash
python codes/main.py --dataset_name BCICIV2A --paradigm cross_session --use_gpu True
```

### Custom Configuration

```bash
python codes/main.py --dataset_name BCICIV2A --dataset_path /path/to/dataset --paradigm cross_session --seed 2023 --gpu_id 0
```

### Hyperparameter Optimization

```bash
python codes/main.py --dataset_name BCICIV2A --is_choose_best_hyper True
```

## Supported Datasets

### BCICIV2A

A motor imagery dataset from the BCI Competition IV, consisting of EEG recordings from 9 subjects performing 4 motor imagery tasks:
- Left hand movement
- Right hand movement
- Foot movement
- Tongue movement

Each subject participated in 2 sessions (training and evaluation) with 288 trials per session.

## Models

### EEG-Conformer

A convolutional transformer architecture specifically designed for EEG decoding. The model combines the local feature extraction capabilities of CNNs with the long-range dependency modeling of transformers.

### EEG-Conformer-CTN

An extension of the EEG-Conformer with additional CTN (Convolutional Transformer Network) components for improved performance.

## Evaluation Paradigms

### Cross-Session

Training on the first session of each subject and testing on the second session of the same subject.

### Cross-Subject

Training on data from multiple subjects and testing on an unseen subject (leave-one-subject-out validation).

## Visualization

The framework provides several visualization tools:

- **Confusion Matrices**: Visualize classification performance across different classes
- **t-SNE**: Visualize the learned feature representations

Example:
```bash
python codes/main.py --draw_list confusion_matrix tsne
```

## Command-line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config-dir` | Directory for config files | `configs` |
| `--seed` | Random seed | `2025` |
| `--dataset_name` | Dataset name | `BCICIV2A` |
| `--dataset_path` | Path to dataset | `D:\dataset\BCICIV_2a_gdf` |
| `--use_org_data` | Use original data | `True` |
| `--paradigm` | Training paradigm | `cross_session` |
| `--augmentation` | Use data augmentation | `True` |
| `--use_gpu` | Use GPU for training | `True` |
| `--gpu_id` | GPU ID | `0` |
| `--log_dir` | Log directory | `./logs` |
| `--save_models_dir` | Model save directory | `./models` |
| `--save_results_dir` | Results save directory | `./results` |
| `--save_model` | Save trained models | `True` |
| `--is_choose_best_hyper` | Use hyperparameter optimization | `False` |
| `--draw_list` | List of visualizations to create | `['confusion_matrix', 'tsne']` |

## Dependencies

- PyTorch
- MNE
- NumPy
- scikit-learn
- wandb
- einops

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The EEG-Conformer implementation is based on [eeyhsong/EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer)
- The MACTN code is based on [ThreePoundUniverse/MACTN](https://github.com/ThreePoundUniverse/MACTN)

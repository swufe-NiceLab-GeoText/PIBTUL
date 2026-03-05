# PIBTUL: Trajectory User Linking

 a novel Prototype-guided Information Bottleneck solution for addressing the TUL task

## Core Features

- **Multi-view Learning**: Original, cropped, and reversed trajectory views
- **Information Bottleneck**: Variational information bottleneck for feature learning
- **Prototype Clustering**: Momentum-updated user prototype classification

## Project Structure

```
PIBTUL/
├── main.py      # Main training script
├── models.py    # Model definitions
├── utils.py     # Data augmentation and utility functions
├── data_load.py # Data loading
└── data/        # Data directory
```

## Quick Start

### Basic Usage
```bash
python main.py
```

### Custom Parameters
```bash
python main.py --batch_size 128 --learning_rate 0.0005 --epochs 80
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embed_size` | 250 | Embedding dimension |
| `hidden_size` | 256 | Hidden layer dimension |
| `batch_size` | 128 | Batch size |
| `learning_rate` | 0.0005 | Learning rate |
| `epochs` | 80 | Training epochs |

## Model Architecture

1. **Data Augmentation**: Random cropping (70%) and sequence reversal
2. **Multi-view Encoding**: View-specific LSTM encoders
3. **Information Bottleneck**: Variational encoding with KL regularization
4. **Classification**: Prototype-based user identification

## Output Files

- Best model: `best_model_{city}_acc{accuracy}_epoch{epoch}.pth`
- Training results: `acc_data_{city}_PIBTUL.json`
- Best metrics: `data/best_results_{city}_PIBTUL.json`


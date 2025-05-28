# Mamba Model Initialization Research

Research on initialization strategies for Mamba-2 state space models, focusing on `dt_bias` parameter initialization and training pipeline.

## Scripts

### `src/init_seeds.py` - Model Initialization
Creates Mamba-2 models with different initialization strategies and random seeds.

**Features:**
- Multiple initialization types: `uniform`, `linear_decay`, `linear_decay_reverse`
- Configurable seeds and parameters
- Batch model creation with different combinations

**Usage:**
```bash
python src/init_seeds.py \
    --config-path ./configs \
    --seeds 42 168 2122 \
    --initializations uniform linear_decay \
    --output-path ./mamba2-40m
```

### `src/train.py` - Model Training
Trains Mamba-2 models using SFTTrainer with comprehensive configuration options.

**Features:**
- SFTTrainer integration with configurable hyperparameters
- Weights & Biases logging support
- Automatic model and tokenizer saving
- CUDA device management

**Usage:**
```bash
python src/train.py \
    --model-path ./mamba2-40m/seed-42-uniform \
    --dataset-path ./openwebtext-tokenized \
    --output-dir ./results/training-run \
    --wandb-token YOUR_TOKEN \
    --batch-size 58 \
    --learning-rate 5e-4
```

### `src/evaluate.py` - Model Evaluation
Evaluates trained Mamba-2 models based on perplexity.

**Features:**
- Calculates perplexity using the `evaluate` library.
- Supports multiple models based on seeds and initialization types.
- Loads datasets from Hugging Face `datasets` library.
- Saves evaluation results to a JSON file.
- Configurable via command-line arguments.

**Usage:**
```bash
python src/evaluate.py \
    --model-base-path ./results \
    --seeds 42 168 2122 \
    --initializations uniform linear_decay \
    --dataset-name wikitext \
    --dataset-config-name wikitext-2-raw-v1 \
    --output-file ./perplexity_results.json
```

## Pipeline
1. **Initialize**: Use `init_seeds.py` to create models with different initialization strategies
2. **Train**: Use `train.py` to train models on your dataset
3. **Evaluate**: Use `src/evaluate.py` to compare performance across different initialization methods

## Model Configuration
- **Size**: ~40M parameters (256 hidden, 12 layers)
- **Sequence Length**: 1024 tokens (configurable)
- **Architecture**: Mamba-2 with configurable `dt_bias` initialization

## Requirements
```bash
pip install -r requirements.txt
```
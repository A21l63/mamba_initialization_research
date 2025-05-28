import os
import argparse
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, Mamba2ForCausalLM
from trl import SFTTrainer, SFTConfig


# Default configuration
DEFAULT_SEED = 2163
DEFAULT_LEARNING_RATE = 5e-4
DEFAULT_BATCH_SIZE = 58
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 17
DEFAULT_MAX_LENGTH = 1024
DEFAULT_NUM_EPOCHS = 1
DEFAULT_WARMUP_RATIO = 0.01


def setup_environment(cuda_visible_devices: str = "0"):
    """Set up the training environment and CUDA settings."""
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    torch.manual_seed(DEFAULT_SEED)


def setup_wandb(wandb_token: str, project_name: str = "huggingface"):
    """Initialize Weights & Biases logging."""
    if wandb_token:
        wandb.login(key=wandb_token)
        print("W&B login successful")
    else:
        print("Warning: No W&B token provided, logging will be disabled")


def load_model_and_tokenizer(model_path: str):
    """
    Load the Mamba2 model and tokenizer from the specified path.
    
    Args:
        model_path: Path to the pretrained model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    try:
        model = Mamba2ForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")


def load_dataset(dataset_path: str):
    """
    Load the preprocessed dataset from disk.
    
    Args:
        dataset_path: Path to the tokenized dataset
        
    Returns:
        Dataset: The loaded dataset with train/test splits
    """
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
        print(f"Successfully loaded dataset from {dataset_path}")
        print(f"Dataset structure: {dataset}")
        return dataset
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from {dataset_path}: {e}")


def create_training_config(
    output_dir: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    seed: int = DEFAULT_SEED,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    logging_steps: int = 1,
    report_to: str = "wandb"
) -> SFTConfig:
    """
    Create training configuration for SFTTrainer.
    
    Args:
        output_dir: Directory to save training outputs
        batch_size: Training batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        warmup_ratio: Warmup ratio for learning rate scheduler
        seed: Random seed
        save_steps: Steps between model saves
        eval_steps: Steps between evaluations
        logging_steps: Steps between logging
        report_to: Logging backend (wandb, tensorboard, etc.)
        
    Returns:
        SFTConfig: Training configuration object
    """
    logs_dir = Path(output_dir).parent / "logs" / Path(output_dir).name
    
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_checkpointing=True,
        logging_dir=str(logs_dir),
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        report_to=report_to,
        lr_scheduler_type='cosine',
        num_train_epochs=num_epochs,
        packing=False,
        eval_strategy="steps",
        save_steps=save_steps,
        save_total_limit=5,
        eval_steps=eval_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        torch_empty_cache_steps=10,
        adam_beta1=0.9,
        adam_beta2=0.99,
        seed=seed,
        warmup_ratio=warmup_ratio,
        bf16=True,
        weight_decay=0.01,
        max_grad_norm=1.0,
        load_best_model_at_end=False
    )
    
    return training_args


def train_model(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    wandb_token: Optional[str] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    warmup_ratio: float = DEFAULT_WARMUP_RATIO,
    seed: int = DEFAULT_SEED,
    save_steps: int = 1000,
    eval_steps: int = 1000,
    save_final_model: bool = True,
    cuda_visible_devices: str = "0"
):
    """
    Train a Mamba2 model using the SFTTrainer.
    
    Args:
        model_path: Path to the pretrained model
        dataset_path: Path to the tokenized dataset
        output_dir: Directory to save training outputs
        wandb_token: Weights & Biases API token
        batch_size: Training batch size per device
        gradient_accumulation_steps: Steps to accumulate gradients
        learning_rate: Learning rate for training
        max_length: Maximum sequence length
        num_epochs: Number of training epochs
        warmup_ratio: Warmup ratio for learning rate scheduler
        seed: Random seed
        save_steps: Steps between model saves
        eval_steps: Steps between evaluations
        save_final_model: Whether to save the final model
        
    Returns:
        SFTTrainer: The trained trainer object
    """
    # Setup environment
    setup_environment(cuda_visible_devices)
    torch.manual_seed(seed)
    
    # Setup W&B if token provided
    if wandb_token:
        setup_wandb(wandb_token)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_path)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_path)
    
    # Create training configuration
    print("Creating training configuration...")
    training_args = create_training_config(
        output_dir=output_dir,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_length=max_length,
        num_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        seed=seed,
        save_steps=save_steps,
        eval_steps=eval_steps,
        report_to="wandb" if wandb_token else "none"
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset.get('test', None),  # Use test set if available
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_length,
    )
    
    # Start training
    print("Starting training...")
    trainer.train()
    
    # Save final model if requested
    if save_final_model:
        final_model_path = f"{output_dir}-final"
        print(f"Saving final model to {final_model_path}")
        trainer.model.save_pretrained(final_model_path)
        trainer.tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")
    
    return trainer


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Train Mamba2 models using SFTTrainer")
    
    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the pretrained model directory")
    parser.add_argument("--dataset-path", type=str, required=True,
                       help="Path to the tokenized dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Directory to save training outputs")
    
    # Optional arguments with defaults
    parser.add_argument("--wandb-token", type=str, default=None,
                       help="Weights & Biases API token")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help="Training batch size per device")
    parser.add_argument("--gradient-accumulation-steps", type=int, 
                       default=DEFAULT_GRADIENT_ACCUMULATION_STEPS,
                       help="Steps to accumulate gradients")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE,
                       help="Learning rate for training")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                       help="Maximum sequence length")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS,
                       help="Number of training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO,
                       help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                       help="Random seed")
    parser.add_argument("--save-steps", type=int, default=1000,
                       help="Steps between model saves")
    parser.add_argument("--eval-steps", type=int, default=1000,
                       help="Steps between evaluations")
    parser.add_argument("--no-save-final", action="store_true",
                       help="Don't save the final model")
    parser.add_argument("--cuda-visible-devices", type=str, default="0",
                       help="CUDA visible devices")
    
    args = parser.parse_args()
    
    # Train the model
    try:
        train_model(
            model_path=args.model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            wandb_token=args.wandb_token,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            num_epochs=args.num_epochs,
            warmup_ratio=args.warmup_ratio,
            seed=args.seed,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            save_final_model=not args.no_save_final,
            cuda_visible_devices=args.cuda_visible_devices
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()

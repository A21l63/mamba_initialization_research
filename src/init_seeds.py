import os
import json
import argparse
from pathlib import Path
from typing import List, Dict

import torch
from transformers import AutoTokenizer, Mamba2Config, Mamba2ForCausalLM
from initializations import initialize_dt_bias


# Default configuration
DEFAULT_SEEDS = [42]
DEFAULT_INITIALIZATIONS = ['uniform', 'linear_decay', 'linear_decay_reverse']

def load_config_from_path(config_path: str) -> Dict:
    """Load model configuration from a config file."""
    config_file = Path(config_path) / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_file}")
    
    with open(config_file, 'r') as f:
        config_dict = json.load(f)
    return config_dict

def create_model_from_config(config_dict: Dict, seed: int) -> Mamba2ForCausalLM:
    """Create a Mamba2 model from configuration dictionary with specified seed."""
    torch.manual_seed(seed)
    
    # Create config object from dictionary
    config = Mamba2Config(**config_dict)
    
    # Create model with the configuration
    model = Mamba2ForCausalLM(config)
    
    return model

def get_tokenizer_from_configs(config_path: str) -> AutoTokenizer:
    """Load tokenizer from the configs directory."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(config_path)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {config_path}: {e}")
        print("Falling back to default tokenizer path")

def create_and_save_model(
    config_path: str,
    seed: int,
    initialization_type: str,
    output_base_path: str = './mamba2-40m',
    t_max: float = 0.1,
    t_min: float = 0.001
):
    """
    Create and save a model with specified seed and initialization.
    
    Args:
        config_path: Path to the configuration directory
        seed: Random seed for model initialization
        initialization_type: Type of initialization ('uniform', 'linear_decay', 'linear_decay_reverse')
        output_base_path: Base path for saving models
        t_max: Maximum time step value for initialization
        t_min: Minimum time step value for initialization
    """
    # Load configuration
    config_dict = load_config_from_path(config_path)
    
    # Create model
    model = create_model_from_config(config_dict, seed)
    
    # Apply initialization
    try:
        initialize_dt_bias(
            model=model,
            t_max=t_max,
            t_min=t_min,
            init_fn_type=initialization_type
        )
        print(f"Applied {initialization_type} initialization to model with seed {seed}")
    except Exception as e:
        print(f"Warning: Could not apply initialization {initialization_type}: {e}")
    
    # Load tokenizer
    tokenizer = get_tokenizer_from_configs(config_path)
    
    # Create output path
    output_path = Path(output_base_path) / f"seed-{seed}-{initialization_type}"
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Saved model to {output_path}")
    
    return model, tokenizer

def initialize_models(
    config_path: str,
    seeds: List[int] = None,
    initializations: List[str] = None,
    output_base_path: str = './mamba2-40m',
    t_max: float = 0.1,
    t_min: float = 0.001,
):
    """
    Initialize models for multiple seeds and initialization types.
    
    Args:
        config_path: Path to the configuration directory
        seeds: List of random seeds to use
        initializations: List of initialization types to apply
        output_base_path: Base path for saving models
        t_max: Maximum time step value for initialization
        t_min: Minimum time step value for initialization
    """
    # Set defaults if not provided
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if initializations is None:
        initializations = DEFAULT_INITIALIZATIONS
    
    # Validate config path
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config path does not exist: {config_path}")
    
    print(f"Initializing models with:")
    print(f"  Config path: {config_path}")
    print(f"  Seeds: {seeds}")
    print(f"  Initializations: {initializations}")
    print(f"  Output path: {output_base_path}")
    
    # Create models for each seed and initialization combination
    created_models = []
    for seed in seeds:
        for init_type in initializations:
            try:
                model, tokenizer = create_and_save_model(
                    config_path=config_path,
                    seed=seed,
                    initialization_type=init_type,
                    output_base_path=output_base_path,
                    t_max=t_max,
                    t_min=t_min
                )
                created_models.append({
                    'seed': seed,
                    'initialization': init_type,
                    'model': model,
                    'tokenizer': tokenizer,
                    'path': Path(output_base_path) / f"seed-{seed}-{init_type}"
                })
                
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                print(f"Error creating model for seed {seed} with {init_type}: {e}")
                continue
    
    print(f"Successfully created {len(created_models)} models")
    return created_models

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Initialize Mamba2 models with different seeds and initializations")
    parser.add_argument("--config-path", type=str, required=True, 
                       help="Path to configuration directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                       help="List of random seeds to use")
    parser.add_argument("--initializations", type=str, nargs="+", default=DEFAULT_INITIALIZATIONS,
                       help="List of initialization types")
    parser.add_argument("--output-path", type=str, default="./mamba2-40m",
                       help="Base output path for saved models")
    parser.add_argument("--t-max", type=float, default=0.1,
                       help="Maximum time step value")
    parser.add_argument("--t-min", type=float, default=0.001,
                       help="Minimum time step value")
    args = parser.parse_args()

    # Initialize models
    created_models = initialize_models(
        config_path=args.config_path,
        seeds=args.seeds,
        initializations=args.initializations,
        output_base_path=args.output_path,
        t_max=args.t_max,
        t_min=args.t_min,
    )
    
    print(f"Initialization complete. Created {len(created_models)} models.")

if __name__ == "__main__":
    main()
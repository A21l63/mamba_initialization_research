import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Mamba2ForCausalLM
from evaluate import load as load_metric # Renamed to avoid conflict with function


# Default configuration
DEFAULT_SEEDS = [42]
DEFAULT_INITIALIZATIONS = ['uniform'] # Example, adjust as needed
DEFAULT_MODEL_BASE_PATH = './mamba2-40m'
DEFAULT_OUTPUT_FILE = 'perplexity_results.json'
DEFAULT_DATASET_NAME = "wikitext"
DEFAULT_DATASET_CONFIG_NAME = "wikitext-2-raw-v1" # Example, adjust as needed
DEFAULT_DATASET_SPLIT = "test"


def load_model_and_tokenizer(model_path: str) -> tuple[Mamba2ForCausalLM, AutoTokenizer]:
    """Load a Mamba2 model and tokenizer from the specified path."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    try:
        model = Mamba2ForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"Successfully loaded model and tokenizer from {model_path}")
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_path}: {e}")

def get_model_paths(
    model_base_path: str,
    seeds: List[int],
    initializations: List[str]
) -> Dict[str, List[str]]:
    """Generate paths to trained models based on seeds and initializations."""
    model_paths = {init_type: [] for init_type in initializations}
    
    for seed in seeds:
        for init_type in initializations:
            # Construct path based on conventions from init_seeds.py and perplexity2.ipynb
            # Example path: ./mamba2-40m/seed-42-uniform/checkpoint-2000
            # Adjust this logic if your path conventions are different
            
            # First, try the structure from init_seeds.py
            path_option1 = Path(model_base_path) / f"seed-{seed}-{init_type}"
            
            # Then, try the structure from perplexity2.ipynb (if different)
            # Example: "mamba2-40m-seed-{seed}-uniform"
            path_option2_base = f"mamba2-40m-seed-{seed}-{init_type.replace('_', '-')}" # replace for linear_decay etc.
            
            # Attempt to find a checkpoint directory, common in training outputs
            # Common checkpoint pattern
            checkpoint_path_option1 = path_option1 / "checkpoint-2000" # Or other common checkpoint numbers
            checkpoint_path_option2 = Path("./results") / path_option2_base / "checkpoint-2000"


            # Check which path exists or has a checkpoint
            # Prioritize checkpoint paths as they usually contain the final/best model
            if checkpoint_path_option1.exists():
                model_paths[init_type].append(str(checkpoint_path_option1))
            elif path_option1.exists():
                 model_paths[init_type].append(str(path_option1))
            elif checkpoint_path_option2.exists():
                 model_paths[init_type].append(str(checkpoint_path_option2))
            elif (Path("./results") / path_option2_base ).exists():
                 model_paths[init_type].append(str(Path("./results") / path_option2_base))
            else:
                print(f"Warning: Model path not found for seed {seed}, init {init_type} at {path_option1} or results/{path_option2_base}")
                
    return model_paths

def load_evaluation_dataset(
    dataset_name: str, 
    dataset_config_name: Optional[str] = None, 
    split: str = "test"
) -> Dataset:
    """Load dataset for evaluation."""
    try:
        if dataset_config_name:
            dataset = load_dataset(dataset_name, dataset_config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded dataset {dataset_name} ({dataset_config_name}) split {split}")
        # Assuming the dataset has a 'text' column as in perplexity2.ipynb
        if 'text' not in dataset.column_names:
            print(f"Warning: 'text' column not found in dataset. Available columns: {dataset.column_names}")
            # Attempt to use the first column if 'text' is not present
            if dataset.column_names:
                print(f"Using column '{dataset.column_names[0]}' as text input.")
                dataset = dataset.rename_column(dataset.column_names[0], 'text')
            else:
                raise ValueError("Dataset has no columns.")

        return dataset['text'] # Return list of text strings
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset_name}: {e}")


def calculate_perplexity(
    model_id: str,
    input_texts: List[str],
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """Calculate perplexity for a given model and input texts."""
    perplexity_metric = load_metric("perplexity", module_type="metric")
    
    # No need to load model and tokenizer here, perplexity.compute handles it
    # model, tokenizer = load_model_and_tokenizer(model_id)

    # Ensure model is on the correct device if loaded manually (though compute should handle it)
    # model.to(device)
    
    print(f"Calculating perplexity for model: {model_id} on device: {device}")

    results = perplexity_metric.compute(
        model_id=model_id,
        add_start_token=False, # As per notebook
        predictions=input_texts,
        batch_size=batch_size,
        device=device 
    )
    
    # Clean up GPU memory if model was loaded manually (not the case here with compute)
    # del model
    # torch.cuda.empty_cache() if device == "cuda" else None
    
    print(f"Perplexity for {model_id}: {results['mean_perplexity']:.4f}")
    return results


def evaluate_models(
    model_base_path: str,
    seeds: List[int],
    initializations: List[str],
    dataset_name: str,
    dataset_config_name: Optional[str],
    dataset_split: str,
    output_file: str,
    batch_size: int = 1,
):
    """
    Evaluate models based on perplexity.
    
    Args:
        model_base_path: Base path where models are stored.
        seeds: List of random seeds used for models.
        initializations: List of initialization types used for models.
        dataset_name: Name of the dataset to use for evaluation (e.g., 'wikitext').
        dataset_config_name: Specific configuration of the dataset (e.g., 'wikitext-2-raw-v1').
        dataset_split: Dataset split to use (e.g., 'test').
        output_file: Path to save the JSON results.
        batch_size: Batch size for perplexity calculation.
    """
    print("Starting model evaluation...")
    
    # Get paths for all models to be evaluated
    all_model_paths_by_type = get_model_paths(model_base_path, seeds, initializations)
    
    # Load evaluation dataset
    print(f"Loading evaluation dataset: {dataset_name} ({dataset_config_name}), split: {dataset_split}")
    input_texts = load_evaluation_dataset(dataset_name, dataset_config_name, dataset_split)
    if not input_texts:
        print("No input texts loaded. Exiting.")
        return

    evaluation_results = {init_type: {'mean': [], 'all_perplexities': [], 'model_paths': []} for init_type in initializations}

    for init_type, model_paths_for_type in all_model_paths_by_type.items():
        if not model_paths_for_type:
            print(f"No models found for initialization type: {init_type}. Skipping.")
            continue

        print(f"\nEvaluating models with '{init_type}' initialization:")
        current_type_perplexities = []
        current_type_means = []

        for model_idx, model_path in enumerate(model_paths_for_type):
            print(f"  Processing model {model_idx + 1}/{len(model_paths_for_type)}: {model_path}")
            try:
                # It's good practice to clear CUDA cache before loading a new model if memory is a concern
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                ppl_results = calculate_perplexity(
                    model_id=model_path,
                    input_texts=input_texts,
                    batch_size=batch_size
                )
                current_type_means.append(ppl_results['mean_perplexity'])
                current_type_perplexities.extend(ppl_results['perplexities']) # Storing all individual perplexities
                evaluation_results[init_type]['model_paths'].append(model_path)

            except FileNotFoundError as e:
                print(f"  Error: Model not found at {model_path}. Skipping. Details: {e}")
            except RuntimeError as e:
                print(f"  Error calculating perplexity for {model_path}. Skipping. Details: {e}")
            except Exception as e:
                print(f"  An unexpected error occurred with {model_path}. Skipping. Details: {e}")
        
        if current_type_means:
            evaluation_results[init_type]['mean'] = current_type_means # list of mean perplexities for each model of this type
            evaluation_results[init_type]['all_perplexities'] = current_type_perplexities # list of all sentence perplexities for this type
            print(f"  Average mean perplexity for '{init_type}': {np.mean(current_type_means):.4f}")
        else:
            print(f"  No perplexity results for '{init_type}'.")


    # Save results to JSON file
    try:
        with open(output_file, 'w') as f:
            json.dump(evaluation_results, f, indent=4)
        print(f"\nEvaluation complete. Results saved to {output_file}")
    except IOError as e:
        print(f"Error saving results to {output_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving results: {e}")
        
    return evaluation_results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Evaluate Mamba2 models using perplexity.")
    
    parser.add_argument("--model-base-path", type=str, default=DEFAULT_MODEL_BASE_PATH,
                        help="Base path where models are stored.")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="List of random seeds for models to evaluate.")
    parser.add_argument("--initializations", type=str, nargs="+", default=DEFAULT_INITIALIZATIONS,
                        help="List of initialization types to evaluate.")
    parser.add_argument("--dataset-name", type=str, default=DEFAULT_DATASET_NAME,
                        help="Name of the Hugging Face dataset for evaluation (e.g., 'wikitext').")
    parser.add_argument("--dataset-config-name", type=str, default=DEFAULT_DATASET_CONFIG_NAME,
                        help="Specific configuration for the dataset (e.g., 'wikitext-2-raw-v1').")
    parser.add_argument("--dataset-split", type=str, default=DEFAULT_DATASET_SPLIT,
                        help="Dataset split to use (e.g., 'test', 'validation').")
    parser.add_argument("--output-file", type=str, default=DEFAULT_OUTPUT_FILE,
                        help="Path to save the JSON results file.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for perplexity calculation.")
    # Add any other arguments from your notebook or other scripts if needed

    args = parser.parse_args()

    print("Configuration:")
    print(f"  Model Base Path: {args.model_base_path}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Initializations: {args.initializations}")
    print(f"  Dataset Name: {args.dataset_name}")
    print(f"  Dataset Config Name: {args.dataset_config_name}")
    print(f"  Dataset Split: {args.dataset_split}")
    print(f"  Output File: {args.output_file}")
    print(f"  Batch Size: {args.batch_size}")

    evaluate_models(
        model_base_path=args.model_base_path,
        seeds=args.seeds,
        initializations=args.initializations,
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        dataset_split=args.dataset_split,
        output_file=args.output_file,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()

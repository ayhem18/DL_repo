import os
import sys
import argparse
import torch

from config import TransformerConfig
from train import train_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a Transformer classifier on synthetic sequence data')
    
    # Data parameters
    parser.add_argument('--max-len', type=int, default=32, help='Maximum sequence length')
    parser.add_argument('--dim', type=int, default=16, help='Feature dimension of each token')
    parser.add_argument('--train-samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--val-samples', type=int, default=2000, help='Number of validation samples')
    parser.add_argument('--test-samples', type=int, default=2000, help='Number of test samples')
    parser.add_argument('--all-same-length', action='store_true', help='Use fixed sequence length')
    parser.add_argument('--max-mean', type=float, default=3.0, help='Maximum mean value for Gaussian distributions')
    parser.add_argument('--data-seed', type=int, default=42, help='Random seed for data generation')
    
    # Model parameters
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--num-heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--key-dim', type=int, default=16, help='Key dimension')
    parser.add_argument('--value-dim', type=int, default=16, help='Value dimension')
    parser.add_argument('--num-transformer-blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--num-classification-layers', type=int, default=2, help='Number of classification layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'max', 'cls', 'last'], 
                        help='Pooling method for sequence representation')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--early-stopping-patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--model-seed', type=int, default=123, help='Random seed for model initialization')
    
    # Logging and output
    parser.add_argument('--log-dir', type=str, default='runs', help='Directory for logs')
    parser.add_argument('--experiment-name', type=str, default='transformer_sanity_check', help='Experiment name')
    parser.add_argument('--no-save-model', action='store_true', help='Do not save model')
    parser.add_argument('--log-interval', type=int, default=10, help='Log interval (batches)')
    
    # Device
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    return parser.parse_args()


def update_config_from_args(config: TransformerConfig, args: argparse.Namespace):
    """Update configuration from command-line arguments."""
    # Data parameters
    config.max_len = args.max_len
    config.dim = args.dim
    config.train_samples = args.train_samples
    config.val_samples = args.val_samples
    config.test_samples = args.test_samples
    config.all_same_length = args.all_same_length
    config.max_mean = args.max_mean
    config.data_seed = args.data_seed
    
    # Model parameters
    config.d_model = args.d_model
    config.num_heads = args.num_heads
    config.key_dim = args.key_dim
    config.value_dim = args.value_dim
    config.num_transformer_blocks = args.num_transformer_blocks
    config.num_classification_layers = args.num_classification_layers
    config.dropout = args.dropout
    config.pooling = args.pooling
    
    # Training parameters
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.early_stopping_patience = args.early_stopping_patience
    config.model_seed = args.model_seed
    
    # Logging and output
    config.log_dir = args.log_dir
    config.experiment_name = args.experiment_name
    config.save_model = not args.no_save_model
    config.log_interval = args.log_interval
    

    return config


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Check if CUDA is available
    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create config and update with args
    config = TransformerConfig()
    # config = update_config_from_args(config, args)
    
    # Print configuration
    print("\nConfiguration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    # Train model
    model, results = train_model(config)
    
    # Print final results
    print("\nFinal Results:")
    print(f"  Train Loss: {results['train_loss']:.4f}, Train Acc: {results['train_acc']:.2f}%")
    print(f"  Val Loss: {results['val_loss']:.4f}, Val Acc: {results['val_acc']:.2f}%")
    print(f"  Test Loss: {results['test_loss']:.4f}, Test Acc: {results['test_acc']:.2f}%")
    print(f"  Total Training Time: {results['total_time']:.2f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

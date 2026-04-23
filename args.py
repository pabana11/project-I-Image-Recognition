import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Object Detection Training Script")
    
    # Hyper Data parameters
    parser.add_argument('--data_dir', type=str, default='./Data/images', help='Directory containing the dataset')
    parser.add_argument('--csv_dir', type=str, default='./Data/CSVs', help='Directory containing the CSV files')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for training')
    
    # Model parameters
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes (including background)')
    
    # Training parameters - MATCHING TRAINER.PY
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate') 
    parser.add_argument('--wd', type=float, default=0.0005, help='Weight decay') .py
    
    # Output parameters
    parser.add_argument('--out_dir', type=str, default='./checkpoints', help='Where to save the model') # Added for trainer.py
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cpu', help='Device to use (cuda or cpu)')
    
    return parser.parse_args()
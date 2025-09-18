"""
Configuration System for Chess Neural Network Training

This module provides configuration management for the chess neural network trainer,
including model architecture, training parameters, and data loading settings.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path


class TrainingConfig:
    """Configuration class for training parameters."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize training configuration.
        
        Args:
            config_dict: Dictionary containing configuration parameters
        """
        # Default configuration
        self.defaults = {
            # Model architecture
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1,
                'input_channels': 18
            },
            
            # Training parameters
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'num_epochs': 100,
                'val_split': 0.1,
                'save_every': 10,
                'early_stopping_patience': 10,
                'optimizer': 'adamw',
                'betas': [0.9, 0.999],
                'gradient_clip_norm': 1.0
            },
            
            # Data loading
            'data': {
                'num_workers': 4,
                'pin_memory': True,
                'max_positions': None,
                'shuffle': True
            },
            
            # Hardware
            'hardware': {
                'device': 'auto',
                'mixed_precision': False
            },
            
            # Paths
            'paths': {
                'data_file': None,
                'save_dir': 'checkpoints',
                'log_dir': 'logs'
            }
        }
        
        # Load configuration
        if config_dict:
            self.config = self._merge_configs(self.defaults, config_dict)
        else:
            self.config = self.defaults.copy()
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'model.d_model')."""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, filepath: str):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()


class ConfigPresets:
    """Predefined configuration presets for different use cases."""
    
    @staticmethod
    def get_small_config() -> TrainingConfig:
        """Small model for quick testing and development."""
        config = {
            'model': {
                'd_model': 128,
                'nhead': 4,
                'num_layers': 3,
                'dim_feedforward': 512,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 16,
                'learning_rate': 2e-4,
                'num_epochs': 20,
                'save_every': 5
            },
            'data': {
                'max_positions': 10000
            }
        }
        return TrainingConfig(config)
    
    @staticmethod
    def get_medium_config() -> TrainingConfig:
        """Medium model for balanced performance and training time."""
        config = {
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'num_epochs': 50
            },
            'data': {
                'max_positions': 100000
            }
        }
        return TrainingConfig(config)
    
    @staticmethod
    def get_large_config() -> TrainingConfig:
        """Large model for maximum performance."""
        config = {
            'model': {
                'd_model': 512,
                'nhead': 16,
                'num_layers': 12,
                'dim_feedforward': 2048,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 64,
                'learning_rate': 5e-5,
                'num_epochs': 100,
                'weight_decay': 1e-4
            },
            'data': {
                'max_positions': None  # Use all available data
            }
        }
        return TrainingConfig(config)
    
    @staticmethod
    def get_fast_training_config() -> TrainingConfig:
        """Configuration optimized for fast training iterations."""
        config = {
            'model': {
                'd_model': 64,
                'nhead': 2,
                'num_layers': 2,
                'dim_feedforward': 256,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 4,
                'learning_rate': 5e-4,
                'num_epochs': 10,
                'save_every': 2
            },
            'data': {
                'max_positions': 5000,
                'num_workers': 2
            }
        }
        return TrainingConfig(config)


class ConfigValidator:
    """Validator for training configurations."""
    
    @staticmethod
    def validate(config: TrainingConfig) -> Dict[str, list]:
        """
        Validate configuration and return any issues found.
        
        Args:
            config: Training configuration to validate
            
        Returns:
            Dictionary of validation issues by category
        """
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Validate model configuration
        model_config = config.get('model', {})
        
        # Check d_model is power of 2
        d_model = model_config.get('d_model', 256)
        if not (d_model & (d_model - 1) == 0):
            issues['warnings'].append(
                f"d_model ({d_model}) is not a power of 2, which may affect performance"
            )
        
        # Check nhead divides d_model
        nhead = model_config.get('nhead', 8)
        if d_model % nhead != 0:
            issues['errors'].append(
                f"nhead ({nhead}) must divide d_model ({d_model})"
            )
        
        # Check dropout is in valid range
        dropout = model_config.get('dropout', 0.1)
        if not 0 <= dropout <= 1:
            issues['errors'].append(
                f"dropout ({dropout}) must be between 0 and 1"
            )
        
        # Validate training configuration
        training_config = config.get('training', {})
        
        # Check batch size is positive
        batch_size = training_config.get('batch_size', 32)
        if batch_size <= 0:
            issues['errors'].append(
                f"batch_size ({batch_size}) must be positive"
            )
        
        # Check learning rate is positive
        learning_rate = training_config.get('learning_rate', 1e-4)
        if learning_rate <= 0:
            issues['errors'].append(
                f"learning_rate ({learning_rate}) must be positive"
            )
        
        # Check val_split is in valid range
        val_split = training_config.get('val_split', 0.1)
        if not 0 < val_split < 1:
            issues['errors'].append(
                f"val_split ({val_split}) must be between 0 and 1"
            )
        
        # Validate data configuration
        data_config = config.get('data', {})
        
        # Check num_workers is non-negative
        num_workers = data_config.get('num_workers', 4)
        if num_workers < 0:
            issues['errors'].append(
                f"num_workers ({num_workers}) must be non-negative"
            )
        
        # Check data file exists if specified
        data_file = config.get('paths.data_file')
        if data_file and not os.path.exists(data_file):
            issues['errors'].append(
                f"Data file does not exist: {data_file}"
            )
        
        return issues
    
    @staticmethod
    def print_validation_results(issues: Dict[str, list]):
        """Print validation results in a formatted way."""
        if issues['errors']:
            print("❌ Configuration Errors:")
            for error in issues['errors']:
                print(f"  - {error}")
        
        if issues['warnings']:
            print("⚠️  Configuration Warnings:")
            for warning in issues['warnings']:
                print(f"  - {warning}")
        
        if not issues['errors'] and not issues['warnings']:
            print("✅ Configuration is valid!")


def create_config_from_args(args) -> TrainingConfig:
    """
    Create configuration from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Training configuration
    """
    config = TrainingConfig()
    
    # Override with command line arguments
    if hasattr(args, 'config_file') and args.config_file:
        config = TrainingConfig.load(args.config_file)
    
    if hasattr(args, 'data_file') and args.data_file:
        config.set('paths.data_file', args.data_file)
    
    if hasattr(args, 'save_dir') and args.save_dir:
        config.set('paths.save_dir', args.save_dir)
    
    if hasattr(args, 'batch_size') and args.batch_size:
        config.set('training.batch_size', args.batch_size)
    
    if hasattr(args, 'learning_rate') and args.learning_rate:
        config.set('training.learning_rate', args.learning_rate)
    
    if hasattr(args, 'num_epochs') and args.num_epochs:
        config.set('training.num_epochs', args.num_epochs)
    
    if hasattr(args, 'max_positions') and args.max_positions:
        config.set('data.max_positions', args.max_positions)
    
    return config


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Test presets
    print("\n1. Testing presets:")
    small_config = ConfigPresets.get_small_config()
    print(f"Small config model d_model: {small_config.get('model.d_model')}")
    
    # Test validation
    print("\n2. Testing validation:")
    issues = ConfigValidator.validate(small_config)
    ConfigValidator.print_validation_results(issues)
    
    # Test save/load
    print("\n3. Testing save/load:")
    test_config_path = "test_config.json"
    small_config.save(test_config_path)
    loaded_config = TrainingConfig.load(test_config_path)
    print(f"Loaded config model d_model: {loaded_config.get('model.d_model')}")
    
    # Cleanup
    os.remove(test_config_path)
    print("Configuration system test completed!")

"""
Transformer-based Neural Network for Chess Position Evaluation

This module implements a transformer architecture specifically designed for chess position evaluation.
The model takes the multi-plane board representation and outputs a win probability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Any, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer to understand board positions."""
    
    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ChessTransformer(nn.Module):
    """
    Transformer-based neural network for chess position evaluation.
    
    Architecture:
    1. Input projection from 8x8x18 to d_model
    2. Positional encoding
    3. Transformer encoder layers
    4. Global pooling and output head
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Extract configuration parameters
        self.d_model = config.get('d_model', 256)
        self.nhead = config.get('nhead', 8)
        self.num_layers = config.get('num_layers', 6)
        self.dim_feedforward = config.get('dim_feedforward', 1024)
        self.dropout = config.get('dropout', 0.1)
        self.input_channels = config.get('input_channels', 18)  # 8x8x18 board representation
        
        # Input projection: 8x8x18 -> 64x1 -> d_model
        self.input_projection = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Flatten(),  # 8x8 -> 64
            nn.Linear(64, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, max_len=64)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()  # Output win probability [0, 1]
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, 18, 8, 8)
            
        Returns:
            Win probability tensor of shape (batch_size, 1)
        """
        batch_size = x.size(0)
        
        # Project input to d_model: (batch_size, 18, 8, 8) -> (batch_size, 64, d_model)
        x = self.input_projection(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)  # (batch_size, 1, d_model)
        
        # Pass through transformer
        x = self.transformer(x)  # (batch_size, 1, d_model)
        
        # Global pooling and output
        x = x.mean(dim=1)  # (batch_size, d_model)
        output = self.output_head(x)  # (batch_size, 1)
        
        return output


class ChessTransformerConfig:
    """Configuration class for ChessTransformer model."""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration for the transformer model."""
        return {
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'dropout': 0.1,
            'input_channels': 18
        }
    
    @staticmethod
    def get_small_config() -> Dict[str, Any]:
        """Get small configuration for faster training/testing."""
        return {
            'd_model': 128,
            'nhead': 4,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1,
            'input_channels': 18
        }
    
    @staticmethod
    def get_large_config() -> Dict[str, Any]:
        """Get large configuration for better performance."""
        return {
            'd_model': 512,
            'nhead': 16,
            'num_layers': 12,
            'dim_feedforward': 2048,
            'dropout': 0.1,
            'input_channels': 18
        }


def create_model(config: Optional[Dict[str, Any]] = None) -> ChessTransformer:
    """
    Factory function to create a ChessTransformer model.
    
    Args:
        config: Model configuration dictionary. If None, uses default config.
        
    Returns:
        Initialized ChessTransformer model
    """
    if config is None:
        config = ChessTransformerConfig.get_default_config()
    
    return ChessTransformer(config)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    config = ChessTransformerConfig.get_default_config()
    model = create_model(config)
    
    # Create dummy input
    dummy_input = torch.randn(2, 18, 8, 8)  # batch_size=2
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model created with {count_parameters(model):,} parameters")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")

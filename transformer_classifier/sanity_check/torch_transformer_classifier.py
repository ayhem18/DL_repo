import torch
from torch import nn
from typing import Optional

from mypt.building_blocks.auxiliary.embeddings.scalar.encoding import PositionalEncoding
from mypt.building_blocks.linear_blocks.fc_blocks import ExponentialFCBlock
from mypt.building_blocks.mixins.general import NonSequentialModuleMixin
from mypt.nets.transformers.pooling_layers import _POOLING_REGISTRY


class PytorchTransformerClassifier(NonSequentialModuleMixin, nn.Module):
    """
    A Transformer-based classifier using PyTorch's native `TransformerEncoder`.

    This class serves as a baseline or alternative to a custom `TransformerClassifier`.
    It uses `torch.nn.TransformerEncoderLayer` and `torch.nn.TransformerEncoder`
    for the main transformer blocks.
    """
    def __init__(
        self,
        d_model: int,
        num_transformer_blocks: int,
        num_classification_layers: int,
        num_heads: int,
        num_classes: int,
        pooling: str = "cls",
        dropout: float = 0.1,
    ):
        """
        Initializes the PytorchTransformerClassifier.

        Args:
            d_model (int): The number of expected features in the input.
            num_transformer_blocks (int): The number of sub-encoder-layers in the encoder.
            num_classification_layers (int): The number of layers in the final classification head.
            num_heads (int): The number of heads in the multiheadattention models.
            num_classes (int): The number of output classes.
            pooling (str): The pooling strategy to use. Defaults to "cls".
            dropout (float): The dropout value. Defaults to 0.1.
        """
        nn.Module.__init__(self)
        NonSequentialModuleMixin.__init__(self, inner_components_fields=['pos_emb', 'encoder', 'pool', 'head'])

        if pooling not in _POOLING_REGISTRY:
            raise ValueError(f"Unknown pooling strategy '{pooling}'.")

        self.pos_emb = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation='gelu',
            norm_first=True,
            batch_first=True  # Assumes (batch, seq, feature) input
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_transformer_blocks
        )

        self.pooling_name = pooling
        self.pool = _POOLING_REGISTRY[pooling]()

        self.head = ExponentialFCBlock(
            output=1 if num_classes == 2 else num_classes,
            in_features=d_model,
            num_layers=num_classification_layers,
            dropout=dropout,
            activation='gelu',
            norm_layer='batchnorm1d'
        )

    def forward(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the classifier.

        Args:
            sequence (torch.Tensor): Input sequence tensor of shape (batch, seq_len, d_model).
            pad_mask (Optional[torch.Tensor]): Padding mask of shape (batch, seq_len),
                                              where `1` indicates a valid token and `0` a pad token.

        Returns:
            torch.Tensor: The output logits from the classification head.
        """
        # Add positional encoding
        # TODO: experiment with pytorch implementation of position encoding !!
        pos_encoding = self.pos_emb(torch.arange(sequence.shape[1], device=sequence.device))
        
        if pad_mask is not None:
            # Mask out positional encoding for padded tokens
            pos_encoding = pos_encoding.masked_fill(~pad_mask.bool().unsqueeze(-1), 0)

        x = sequence + pos_encoding

        # PyTorch's TransformerEncoder expects a boolean mask where True means "ignore".
        # The provided pad_mask has 1 for valid tokens, 0 for padding. We need to invert it.
        src_key_padding_mask = None

        if pad_mask is not None:
            src_key_padding_mask = (pad_mask == 0)

        x = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        # Apply the pooling layer
        pooled_output = self.pool(x, pad_mask)

        # Pass through the classification head
        result = self.head(pooled_output)
        
        return result

    def get_pooling_type(self) -> str:
        """Returns the pooling strategy name."""
        return self.pooling_name

    def __call__(self, sequence: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(sequence, pad_mask)

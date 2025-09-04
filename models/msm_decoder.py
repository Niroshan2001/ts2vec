import torch
import torch.nn as nn
import torch.nn.functional as F

class MSMDecoder(nn.Module):
    """
    Masked Signal Modeling (MSM) Decoder for reconstructing masked time series segments.
    Following the MAE design philosophy - lightweight decoder for efficiency.
    """
    
    def __init__(self, input_dims, hidden_dims=64, depth=3):
        super().__init__()
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        
        # Lightweight decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Linear(input_dims if i == 0 else hidden_dims, hidden_dims)
            for i in range(depth)
        ])
        
        # Final reconstruction layer
        self.reconstruction_head = nn.Linear(hidden_dims, input_dims)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, encoded_repr, mask):
        """
        Args:
            encoded_repr: Encoded representations (B x T x output_dims)
            mask: Boolean mask indicating which positions were masked (B x T)
        
        Returns:
            reconstructed: Reconstructed values for masked positions (B x T x input_dims)
        """
        x = encoded_repr
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = F.relu(layer(x))
            x = self.dropout(x)
        
        # Final reconstruction
        reconstructed = self.reconstruction_head(x)
        
        return reconstructed

class MSMLoss(nn.Module):
    """
    Masked Signal Modeling Loss for reconstruction objective.
    """
    
    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, reconstructed, target, mask):
        """
        Args:
            reconstructed: Reconstructed values (B x T x input_dims)
            target: Original target values (B x T x input_dims)
            mask: Boolean mask indicating which positions were masked (B x T)
        
        Returns:
            loss: MSM reconstruction loss
        """
        # Only compute loss on masked positions
        mask = mask.unsqueeze(-1).expand_as(target)  # B x T x input_dims
        
        if self.loss_type == 'mse':
            loss = F.mse_loss(reconstructed[~mask], target[~mask], reduction='mean')
        elif self.loss_type == 'mae':
            loss = F.l1_loss(reconstructed[~mask], target[~mask], reduction='mean')
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")
        
        return loss

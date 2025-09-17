import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder
from models.losses import hierarchical_contrastive_loss
from models.msm_decoder import MSMDecoder, MSMLoss
from utils import take_per_row, split_with_nan, centerize_vary_length_series, torch_pad_nan
import math

class TS2VecMSM:
    '''The TS2Vec-MSM hybrid model with contrastive learning and masked signal modeling'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None,
        # New MSM parameters
        msm_weight=0.5,  # λ parameter for balancing losses
        msm_mask_rate=0.15,  # Percentage of tokens to mask
        msm_decoder_depth=3,
        dynamic_lambda=False  # Whether to use dynamic λ scheduling
    ):
        ''' Initialize a TS2Vec-MSM model.
        
        Args:
            input_dims (int): The input dimension. For a univariate time series, this should be set to 1.
            output_dims (int): The representation dimension.
            hidden_dims (int): The hidden dimension of the encoder.
            depth (int): The number of hidden residual blocks in the encoder.
            device (int): The gpu used for training and inference.
            lr (int): The learning rate.
            batch_size (int): The batch size.
            max_train_length (Union[int, NoneType]): The maximum allowed sequence length for training.
            temporal_unit (int): The minimum unit to perform temporal contrast.
            after_iter_callback (Union[Callable, NoneType]): A callback function after each iteration.
            after_epoch_callback (Union[Callable, NoneType]): A callback function after each epoch.
            msm_weight (float): Weight for MSM loss (λ parameter, 0=contrastive only, 1=MSM only).
            msm_mask_rate (float): Percentage of timestamps to mask for MSM.
            msm_decoder_depth (int): Number of layers in the MSM decoder.
            dynamic_lambda (bool): Whether to use dynamic λ scheduling during training.
        '''
        
        super().__init__()
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.max_train_length = max_train_length
        self.temporal_unit = temporal_unit
        self.msm_weight = msm_weight
        self.msm_mask_rate = msm_mask_rate
        self.dynamic_lambda = dynamic_lambda
        self.input_dims = input_dims
        
        # TS2Vec encoder (discriminative)
        self._net = TSEncoder(
            input_dims=input_dims, 
            output_dims=output_dims, 
            hidden_dims=hidden_dims, 
            depth=depth
        ).to(self.device)
        
        # MSM decoder (generative)
        self._msm_decoder = MSMDecoder(
            encoder_dims=output_dims,    # Takes encoder output (e.g., 320)
            target_dims=input_dims,      # Reconstructs original signal (e.g., 1)
            hidden_dims=hidden_dims,
            depth=msm_decoder_depth
        ).to(self.device)
        
        # Loss functions
        self._msm_loss = MSMLoss(loss_type='mse')
        
        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 0
        self.n_iters = 0
        self.training = True  # Add training flag
        
    def _get_dynamic_lambda(self, epoch, total_epochs):
        """
        Dynamic λ scheduling: start with contrastive learning, gradually add MSM
        
        Multiple scheduling strategies available:
        1. Linear: Start at 0, linearly increase to target lambda
        2. Exponential: Slow start, then rapid increase
        3. Step: Fixed periods of different lambdas
        4. Cosine: Smooth transition with cosine annealing
        """
        if not self.dynamic_lambda:
            return self.msm_weight
        
        if total_epochs is None or total_epochs == 0:
            total_epochs = 100  # Default fallback
            
        progress = min(epoch / total_epochs, 1.0)  # Clamp to [0, 1]
        target_lambda = self.msm_weight
        
        # STRATEGY 1: Linear warmup (recommended for TS2Vec-MSM)
        # Start with pure contrastive (λ=0), linearly increase to target
        warmup_epochs = int(0.3 * total_epochs)  # 30% of training for warmup
        
        if epoch < warmup_epochs:
            # Linear warmup from 0 to target_lambda
            current_lambda = target_lambda * (epoch / warmup_epochs)
        else:
            # Stay at target lambda
            current_lambda = target_lambda
            
        # STRATEGY 2: Alternative - Exponential warmup (uncomment to use)
        # current_lambda = target_lambda * (1 - math.exp(-3 * progress))
        
        # STRATEGY 3: Alternative - Step schedule (uncomment to use)
        # if progress < 0.2:
        #     current_lambda = 0.0          # Pure contrastive for first 20%
        # elif progress < 0.5:
        #     current_lambda = target_lambda * 0.3   # Light MSM for next 30%
        # else:
        #     current_lambda = target_lambda          # Full MSM for last 50%
        
        return current_lambda
    
    def _generate_msm_mask(self, batch_size, seq_len):
        """
        Generate masks for Masked Signal Modeling.
        Supports both random and block masking strategies.
        """
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=self.device)
        
        for i in range(batch_size):
            # Random masking strategy
            n_mask = int(seq_len * self.msm_mask_rate)
            mask_indices = torch.randperm(seq_len)[:n_mask]
            mask[i, mask_indices] = False
            
        return mask
    
    def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
        ''' Training the TS2Vec-MSM model.
        
        Args:
            train_data (numpy.ndarray): The training data. It should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            n_epochs (Union[int, NoneType]): The number of epochs. When this reaches the maximum, the training stops.
            n_iters (Union[int, NoneType]): The number of iterations. When this reaches the maximum, the training stops. If both n_epochs and n_iters are not specified, a default setting would be used that sets n_iters to 200 for a dataset with size <= 100000, 600 otherwise.
            verbose (bool): Whether to print the training loss after each epoch.
            
        Returns:
            loss_log: a list containing the training losses on each epoch.
        '''
        assert train_data.ndim == 3
        
        if n_iters is None and n_epochs is None:
            n_iters = 200 if train_data.size <= 100000 else 600  # default param for n_iters
            
        if self.max_train_length is not None:
            sections = train_data.shape[1] // self.max_train_length
            if sections >= 2:
                train_data = train_data[:, :self.max_train_length * sections]
                train_data = train_data.reshape(train_data.shape[0] * sections, self.max_train_length, train_data.shape[2])
        
        temporal_missing = np.isnan(train_data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            train_data = centerize_vary_length_series(train_data)
            
        train_data = train_data[~np.isnan(train_data).all(axis=2).all(axis=1)]
        
        train_dataset = TensorDataset(torch.from_numpy(train_data).to(torch.float))
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True, drop_last=True)
        
        optimizer = torch.optim.AdamW(
            list(self._net.parameters()) + list(self._msm_decoder.parameters()), 
            lr=self.lr
        )
        
        loss_log = []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            cum_contrastive_loss = 0
            cum_msm_loss = 0
            n_epoch_iters = 0
            
            interrupted = False
            for batch in train_loader:
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                
                x = batch[0]
                if self.max_train_length is not None and x.size(1) > self.max_train_length:
                    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
                    x = x[:, window_offset : window_offset + self.max_train_length]
                x = x.to(self.device)
                
                # Get current λ for this epoch
                current_lambda = self._get_dynamic_lambda(self.n_epochs, n_epochs or 100)
                
                # Log lambda changes (only at the beginning of each epoch)
                if n_epoch_iters == 0 and self.dynamic_lambda and verbose:
                    print(f"Epoch {self.n_epochs + 1}: λ = {current_lambda:.4f}")
                
                
                optimizer.zero_grad()
                
                # Create two different views for contrastive learning (same as TS2Vec)
                ts_l = x.size(1)
                crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
                crop_left = np.random.randint(ts_l - crop_l + 1)
                crop_right = crop_left + crop_l
                crop_eleft = np.random.randint(crop_left + 1)
                crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
                crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
                
                # Forward pass through encoder for contrastive learning
                out1 = self._net(take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
                out1 = out1[:, -crop_l:]
                
                out2 = self._net(take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
                out2 = out2[:, :crop_l]
                
                # Contrastive loss (discriminative objective)
                contrastive_loss = hierarchical_contrastive_loss(
                    out1,
                    out2,
                    temporal_unit=self.temporal_unit
                )
                
                # MSM loss (generative objective)
                # Use full sequence for MSM
                msm_mask = self._generate_msm_mask(x.size(0), x.size(1))
                full_out = self._net(x)
                reconstructed = self._msm_decoder(full_out, msm_mask)
                msm_loss = self._msm_loss(reconstructed, x, msm_mask)
                
                # Combined loss
                total_loss = (1 - current_lambda) * contrastive_loss + current_lambda * msm_loss
                
                total_loss.backward()
                optimizer.step()
                
                # Clear unused memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                cum_loss += total_loss.item()
                cum_contrastive_loss += contrastive_loss.item()
                cum_msm_loss += msm_loss.item()
                n_epoch_iters += 1
                
                self.n_iters += 1
                
                if self.after_iter_callback is not None:
                    self.after_iter_callback(self, total_loss.item())
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            cum_contrastive_loss /= n_epoch_iters
            cum_msm_loss /= n_epoch_iters
            loss_log.append(cum_loss)
            
            if verbose:
                print(f"Epoch #{self.n_epochs}: loss={cum_loss:.6f} "
                      f"(contrastive={cum_contrastive_loss:.6f}, "
                      f"msm={cum_msm_loss:.6f}, λ={current_lambda:.3f})")
            
            self.n_epochs += 1
            
            if self.after_epoch_callback is not None:
                self.after_epoch_callback(self, cum_loss)
        
        return loss_log
    
    def _eval_with_pooling(self, x, mask=None, slicing=None, encoding_window=None):
        """Helper method for encoding with pooling (same as original TS2Vec)"""
        out = self._net(x.to(self.device, non_blocking=True), mask)
        if encoding_window == 'full_series':
            if slicing is not None:
                out = out[:, slicing]
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = out.size(1),
            ).transpose(1, 2)
            
        elif isinstance(encoding_window, int):
            out = F.max_pool1d(
                out.transpose(1, 2),
                kernel_size = encoding_window,
                stride = 1,
                padding = encoding_window // 2
            ).transpose(1, 2)
            if encoding_window % 2 == 0:
                out = out[:, :-1]
            if slicing is not None:
                out = out[:, slicing]
            
        elif encoding_window == 'multiscale':
            p = 0
            reprs = []
            while (1 << p) + 1 < out.size(1):
                t_out = F.max_pool1d(
                    out.transpose(1, 2),
                    kernel_size = (1 << (p + 1)) + 1,
                    stride = 1,
                    padding = 1 << p
                ).transpose(1, 2)
                if slicing is not None:
                    t_out = t_out[:, slicing]
                reprs.append(t_out)
                p += 1
            out = torch.cat(reprs, dim=-1)
            
        else:
            if slicing is not None:
                out = out[:, slicing]
                
        return out.cpu()
    
    def encode(self, data, mask='all_true', encoding_window=None, causal=False, sliding_length=None, sliding_padding=0, batch_size=None):
        ''' Compute representations for the given data.
        
        Args:
            data (numpy.ndarray): This should have a shape of (n_instance, n_timestamps, n_features). All missing data should be set to NaN.
            mask (str): The mask used by encoder can be set to 'binomial', 'continuous', 'all_true', 'all_false', 'mask_last'.
            encoding_window (Union[str, int]): When this param is specified, the computed representation would the max pooling over this window. This can be set to 'full_series', 'multiscale' or an integer specifying the pooling kernel size.
            causal (bool): When this param is set to True, the future informations would not be encoded into representation of each timestamp.
            sliding_length (Union[int, NoneType]): The length of sliding window. When this param is specified, a sliding inference would be applied on the time series.
            sliding_padding (int): This param specifies the contextual data length used for inference every sliding windows.
            batch_size (Union[int, NoneType]): The batch size used for inference. If not specified, this would be the same batch size as training.
            
        Returns:
            repr: The representations for data.
        '''
        # Use the same encoding logic as original TS2Vec
        assert self.training == False, "Model must be in eval mode for encoding"
        assert data.ndim == 3
        
        if batch_size is None:
            batch_size = min(self.batch_size, 4)  # Reduce batch size for evaluation to save memory
        
        n_samples, ts_l, _ = data.shape
        
        org_training = self.training
        self.eval()
        
        dataset = TensorDataset(torch.from_numpy(data).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0]
                
                # Clear CUDA cache between batches to prevent OOM
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if sliding_length is not None:
                    reprs = []
                    if n_samples < batch_size:
                        calc_buffer = []
                        calc_buffer.append(x)
                        while len(calc_buffer) * n_samples < batch_size:
                            calc_buffer.append(x)
                        x = torch.cat(calc_buffer, dim=0)
                    
                    if sliding_padding > 0:
                        x = F.pad(x, (0, 0, sliding_padding, sliding_padding), mode='constant', value=np.nan)
                    
                    # Process sliding windows in smaller chunks
                    chunk_size = min(10, x.size(1) - sliding_length + 1)  # Process 10 windows at a time
                    for start_i in range(0, x.size(1) - sliding_length + 1, chunk_size):
                        end_i = min(start_i + chunk_size, x.size(1) - sliding_length + 1)
                        chunk_reprs = []
                        
                        for i in range(start_i, end_i):
                            out = self._eval_with_pooling(
                                x[:, i : i + sliding_length].to(self.device), 
                                mask,
                                encoding_window=encoding_window
                            )
                            chunk_reprs.append(out[:n_samples])  # Already moved to CPU in _eval_with_pooling
                        
                        reprs.extend(chunk_reprs)
                        
                        # Clear cache after each chunk
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    
                    out = torch.stack(reprs, dim=1)
                    if encoding_window == 'full_series':
                        out = F.max_pool1d(
                            out.transpose(1, 2).contiguous(),
                            kernel_size = out.size(1),
                        ).squeeze(1)
                else:
                    out = self._eval_with_pooling(x, mask, encoding_window=encoding_window)
                    if encoding_window == 'full_series':
                        out = out.squeeze(1)
                        
                output.append(out)
                
                # Clear CUDA cache after each batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
        output = torch.cat(output, dim=0)
        
        self.train(org_training)
        return output.numpy()
    
    def save(self, fn):
        ''' Save the model to a file.
        
        Args:
            fn (str): filename.
        '''
        torch.save({
            'encoder': self._net.state_dict(),
            'decoder': self._msm_decoder.state_dict(),
            'config': {
                'input_dims': self.input_dims,
                'output_dims': self._net.output_dims,
                'hidden_dims': self._net.hidden_dims,
                'msm_weight': self.msm_weight,
                'msm_mask_rate': self.msm_mask_rate
            }
        }, fn)
    
    def load(self, fn):
        ''' Load the model from a file.
        
        Args:
            fn (str): filename.
        '''
        checkpoint = torch.load(fn, map_location=self.device)
        self._net.load_state_dict(checkpoint['encoder'])
        self._msm_decoder.load_state_dict(checkpoint['decoder'])
        
    def eval(self):
        """Set model to evaluation mode"""
        self._net.eval()
        self._msm_decoder.eval()
        self.training = False
        
    def train(self, mode=True):
        """Set model to training mode"""
        self._net.train(mode)
        self._msm_decoder.train(mode)
        self.training = mode

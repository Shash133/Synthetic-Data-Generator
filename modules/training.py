import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.rand(real_samples.size(0), 1)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class EarlyStopping:
    """Early stopping implementation"""
    def __init__(self, patience=50, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_gan(generator, discriminator, data_tensor, config, progress_callback=None):
    """
    Train the GAN model
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        data_tensor: Training data tensor
        config: Configuration object
        progress_callback: Callback function for progress updates
        
    Returns:
        Tuple of (g_losses, d_losses)
    """
    optimizer_G = optim.Adam(generator.parameters(), lr=config.LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.LR, betas=(0.5, 0.9))
    
    early_stopper = EarlyStopping(patience=50)
    g_losses, d_losses = [], []
    
    for epoch in range(config.EPOCHS):
        # Train discriminator
        for _ in range(config.CRITIC_ITER):
            idx = torch.randint(0, len(data_tensor), (config.BATCH_SIZE,))
            real_batch = data_tensor[idx]
            
            z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM)
            fake_batch = generator(z)
            
            real_loss = -torch.mean(discriminator(real_batch))
            fake_loss = torch.mean(discriminator(fake_batch.detach()))
            gp = compute_gradient_penalty(discriminator, real_batch, fake_batch)
            d_loss = real_loss + fake_loss + config.LAMBDA_GP * gp
            
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()
        
        # Train generator
        z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM)
        g_loss = -torch.mean(discriminator(generator(z)))
        
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # Update progress if callback provided
        if progress_callback and epoch % 10 == 0:
            progress_callback(epoch, config.EPOCHS, g_loss.item(), d_loss.item())
        
        # Check for early stopping
        if epoch % 50 == 0:
            if early_stopper(g_loss.item()):
                break
    
    return g_losses, d_losses

def generate_synthetic_samples(n_samples, generator, num_scaler, value_ranges, config):
    """
    Generate synthetic wine samples
    
    Args:
        n_samples: Number of samples to generate
        generator: Trained generator model
        num_scaler: Numerical feature scaler
        value_ranges: Value ranges for features
        config: Configuration object
        
    Returns:
        Tuple of (features, target)
    """
    with torch.no_grad():
        z = torch.randn(n_samples, config.LATENT_DIM)
        synthetic = generator(z).numpy()
        
        # Split numerical and categorical parts
        synthetic_features = synthetic[:, :config.NUM_FEATURES]
        synthetic_target = synthetic[:, config.NUM_FEATURES:]
        
        # Process features
        features = num_scaler.inverse_transform(synthetic_features)
        for i in range(features.shape[1]):
            features[:, i] = np.clip(features[:, i],
                                   value_ranges['min'][i],
                                   value_ranges['max'][i])
            features[:, i] = np.abs(features[:, i])
        
        # Process target
        target = np.argmax(synthetic_target, axis=1) + 3  # Quality ranges from 3-8
        
        return features, target

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.HIDDEN_DIM),
            nn.LayerNorm(config.HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM*2),
            nn.LayerNorm(config.HIDDEN_DIM*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(config.HIDDEN_DIM*2, config.NUM_FEATURES + sum(config.CAT_DIMS)),
        )
        self.num_features = config.NUM_FEATURES
        self.cat_dims = config.CAT_DIMS

    def forward(self, z):
        output = self.model(z)
        num_output = torch.sigmoid(output[:, :self.num_features])
        cat_output = output[:, self.num_features:]
        
        # Process categorical outputs
        cat_probs = torch.softmax(cat_output, dim=1)
        
        return torch.cat([num_output, cat_probs], dim=1)

class Discriminator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(config.NUM_FEATURES + sum(config.CAT_DIMS), config.HIDDEN_DIM*2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(config.HIDDEN_DIM*2, config.HIDDEN_DIM),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(config.HIDDEN_DIM, 1)
        )

    def forward(self, x):
        return self.model(x)

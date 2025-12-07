import torch
import torch.optim as optim
from tqdm import tqdm

class SemidiscreteOT_Solver:
    """
    SD-OTの双対変数(ポテンシャル) g を学習するソルバー。
    """
    def __init__(self, dataset_size, feature_dim, device, batch_size_noise=1024, lr=0.1, ema_decay=0.99):
        self.M = dataset_size
        self.D = feature_dim
        self.device = device
        self.batch_size_noise = batch_size_noise
        
        self.g = torch.zeros(self.M, device=device, requires_grad=True)
        self.g_ema = torch.zeros(self.M, device=device, requires_grad=False)
        self.ema_decay = ema_decay
        self.optimizer = optim.SGD([self.g], lr=lr)
        self.target_weights = torch.full((self.M,), 1.0/self.M, device=device)

    def train_loop(self, flattened_dataset_tensor, num_iterations=20000):
        self.g.requires_grad_(False)
        Y = flattened_dataset_tensor.to(self.device)
        Y_sq_norm = torch.sum(Y ** 2, dim=1) # [M]

        pbar = tqdm(range(num_iterations), desc="[Phase 1] Learning SD-OT Potential")
        for _ in pbar:
            X = torch.randn(self.batch_size_noise, self.D, device=self.device)
            
            # Score = g_j - ||y_j||^2 + 2<x, y_j>
            cross_term = 2 * torch.matmul(X, Y.t())
            bias = self.g - Y_sq_norm
            scores = cross_term + bias.unsqueeze(0) # [N, M]
            
            indices = torch.argmax(scores, dim=1)
            
            hits = torch.zeros(self.M, device=self.device)
            hits.scatter_add_(0, indices, torch.ones(self.batch_size_noise, device=self.device))
            
            grad = (hits / self.batch_size_noise) - self.target_weights
            
            if self.g.grad is None: self.g.grad = torch.zeros_like(self.g)
            self.g.grad.copy_(-grad) # Maximize
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            with torch.no_grad():
                self.g_ema.mul_(self.ema_decay).add_(self.g, alpha=1 - self.ema_decay)

        return self.g_ema
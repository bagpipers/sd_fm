import torch
import torch.optim as optim
from tqdm import tqdm

class SemidiscreteOT_Solver:
    """
    SD-OTの双対変数(ポテンシャル) g を学習するソルバー。
    特徴量(PCA済み想定)を受け取り、Chunkingを用いて省メモリで学習する。
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

    def train_loop(self, flattened_dataset_tensor, num_iterations=20000, chunk_size=10000):
        self.g.requires_grad_(False) # マニュアルで勾配計算するためFalse開始
        Y = flattened_dataset_tensor.to(self.device)
        Y_sq_norm = torch.sum(Y ** 2, dim=1) # [M]

        pbar = tqdm(range(num_iterations), desc="[Phase 1] Learning SD-OT Potential")
        
        for _ in pbar:
            X = torch.randn(self.batch_size_noise, self.D, device=self.device)
            indices = self._get_argmax_indices_chunked(X, Y, Y_sq_norm, chunk_size)
            hits = torch.zeros(self.M, device=self.device)
            hits.scatter_add_(0, indices, torch.ones(self.batch_size_noise, device=self.device))
            grad = (hits / self.batch_size_noise) - self.target_weights
            if self.g.grad is None: 
                self.g.grad = torch.zeros_like(self.g)
            self.g.grad.copy_(-grad) 
            
            self.g.requires_grad_(True) # stepのためにTrue
            self.optimizer.step()
            self.g.requires_grad_(False)
            self.optimizer.zero_grad()
            with torch.no_grad():
                self.g_ema.mul_(self.ema_decay).add_(self.g, alpha=1 - self.ema_decay)

        return self.g_ema

    def _get_argmax_indices_chunked(self, X, Y, Y_sq_norm, chunk_size):
        M = self.M
        B = X.shape[0]
        
        best_scores = torch.full((B,), float('-inf'), device=self.device)
        best_indices = torch.zeros((B,), dtype=torch.long, device=self.device)
        
        for i in range(0, M, chunk_size):
            end = min(i + chunk_size, M)
            
            Y_chunk = Y[i:end]
            g_chunk = self.g[i:end]
            Y_sq_chunk = Y_sq_norm[i:end]
            
            cross_term = 2 * torch.matmul(X, Y_chunk.t())
            bias = g_chunk 
            scores = cross_term + bias.unsqueeze(0)
            
            chunk_max_scores, chunk_max_indices = torch.max(scores, dim=1)
            
            mask = chunk_max_scores > best_scores
            best_scores[mask] = chunk_max_scores[mask]
            best_indices[mask] = chunk_max_indices[mask] + i
            
        return best_indices
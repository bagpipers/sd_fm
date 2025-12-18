import torch
import torch.optim as optim
import os
from tqdm import tqdm

class SemidiscreteOT_Solver:
    """
    SD-OTの双対変数(ポテンシャル) g を学習するソルバー。
    特徴量(PCA済み想定)を受け取り、Chunkingを用いて省メモリで学習する。
    
 
    論文 (arXiv:2509.25519) 
    コスト関数を内積 c(x, y) = -x^T y として実装。
    最大化対象のスコアは S(x, y_j) = x^T y_j + g_j となる。
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

    def save_checkpoint(self, path, step):
        """学習状態を保存"""
        torch.save({
            'step': step,
            'g': self.g.data, # Tensor data only
            'g_ema': self.g_ema.data,
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path):
        """学習状態を復元"""
        if not os.path.exists(path):
            return 0
        
        print(f"Loading SD-OT checkpoint from {path}...")
        ckpt = torch.load(path, map_location=self.device)
        
        self.g.data.copy_(ckpt['g'])
        self.g_ema.data.copy_(ckpt['g_ema'])
        
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
            
        return ckpt.get('step', 0)

    def train_loop(self, flattened_dataset_tensor, num_iterations=20000, chunk_size=10000, checkpoint_path=None, save_interval=10000):
        """
        Args:
            flattened_dataset_tensor (torch.Tensor): CPU上のTensor (mmap推奨)。GPUには転送しないこと！
            checkpoint_path (str): チェックポイント保存先のパス。
        """
        self.g.requires_grad_(False)
        Y_cpu = flattened_dataset_tensor 
        
        start_step = 0
        if checkpoint_path is not None:
            start_step = self.load_checkpoint(checkpoint_path)
            if start_step >= num_iterations:
                print("SD-OT Potential training already finished.")
                return self.g_ema
        pbar = tqdm(range(start_step, num_iterations), desc="[Phase 1] Learning SD-OT Potential", initial=start_step, total=num_iterations)
        
        for step in pbar:
            X = torch.randn(self.batch_size_noise, self.D, device=self.device)
            indices = self._get_argmax_indices_chunked(X, Y_cpu, chunk_size)
            hits = torch.zeros(self.M, device=self.device)
            hits.scatter_add_(0, indices, torch.ones(self.batch_size_noise, device=self.device))
            grad = (hits / self.batch_size_noise) - self.target_weights
            if self.g.grad is None: 
                self.g.grad = torch.zeros_like(self.g)
            self.g.grad.copy_(grad) 
            
            self.g.requires_grad_(True) 
            self.optimizer.step()
            self.g.requires_grad_(False)
            self.optimizer.zero_grad()
            with torch.no_grad():
                self.g_ema.mul_(self.ema_decay).add_(self.g, alpha=1 - self.ema_decay)
            if checkpoint_path is not None and (step + 1) % save_interval == 0:
                self.save_checkpoint(checkpoint_path, step + 1)
        if checkpoint_path is not None:
            self.save_checkpoint(checkpoint_path, num_iterations)

        return self.g_ema

    def _get_argmax_indices_chunked(self, X, Y_cpu, chunk_size):
        """
        メモリ節約版 MIPS探索。
        Y_cpu は CPU Tensor。ループ内で必要な部分だけ GPU に転送して計算する。
        """
        M = self.M
        B = X.shape[0]
        best_scores = torch.full((B,), float('-inf'), device=self.device)
        best_indices = torch.zeros((B,), dtype=torch.long, device=self.device)
        
        for start_idx in range(0, M, chunk_size):
            end_idx = min(start_idx + chunk_size, M)
            Y_chunk = Y_cpu[start_idx:end_idx].to(self.device, non_blocking=True) # [Chunk, D]
            g_chunk = self.g[start_idx:end_idx]   # gは元々GPUにある [Chunk]
            cross_term = torch.matmul(X, Y_chunk.t()) # [B, Chunk]
            scores = cross_term + g_chunk.unsqueeze(0)
            chunk_max_scores, chunk_arg_max = torch.max(scores, dim=1)
            update_mask = chunk_max_scores > best_scores
            best_scores[update_mask] = chunk_max_scores[update_mask]
            best_indices[update_mask] = chunk_arg_max[update_mask] + start_idx
            
        return best_indices
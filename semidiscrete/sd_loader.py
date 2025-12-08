import torch
from torch.utils.data import IterableDataset

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM用データセット。
    1. ランダムノイズ x を生成 (Original Space)
    2. PCA射影 (x -> x_pca)
    3. チャンク分割して MIPS (Maximum Inner Product Search) を行い、最適なインデックス k を決定
    4. オリジナルデータセットから k 番目の (画像, プロンプト) を取得して返す
    """
    def __init__(self, 
                 original_dataset, 
                 potential_g, 
                 dataset_features, 
                 feature_dim, 
                 device, 
                 batch_size=64, 
                 pca_proj=None, 
                 pca_mean=None,
                 chunk_size=10000):
        
        self.dataset = original_dataset
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dataset_features = dataset_features.to(device) 
        self.g = potential_g.to(device)
        
        self.pca_proj = pca_proj.to(device) if pca_proj is not None else None
        self.pca_mean = pca_mean.to(device) if pca_mean is not None else None
        
        if self.pca_proj is not None:
            self.noise_dim = self.pca_proj.shape[0] 
        else:
            self.noise_dim = feature_dim

    def __iter__(self):
        while True:
            X_raw = torch.randn(self.batch_size, self.noise_dim, device=self.device)
            if self.pca_proj is not None:
                X_feat = torch.matmul(X_raw - self.pca_mean, self.pca_proj)
            else:
                X_feat = X_raw
            indices = self._get_indices_chunked(X_feat)
            
            batch_pixels = []
            batch_pos = []
            batch_neg = []
            
            for idx in indices:
                item = self.dataset[int(idx)]
                batch_pixels.append(item["image"]) 
                
                batch_pos.append(item["positive_prompt"])
                batch_neg.append(item["negative_prompt"])
                
            yield {
                "pixel_values": torch.stack(batch_pixels), # x_1 (Target Image)
                "noise": X_raw,                            # x_0 (Paired Noise - Raw)
                "positive_prompt": batch_pos,
                "negative_prompt": batch_neg
            }

    def _get_indices_chunked(self, X_feat):
        """
        メモリ節約のため、データセット側(Y)をチャンク分割して最大スコアを探索する。
        X_feat: [B, D]
        Score = x^T y + g (内積コスト)
        """
        N = self.dataset_features.shape[0]
        B = X_feat.shape[0]
        
        best_scores = torch.full((B,), float('-inf'), device=self.device)
        best_indices = torch.zeros((B,), dtype=torch.long, device=self.device)
        
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            
            Y_chunk = self.dataset_features[i:end]   # [Chunk, D]
            g_chunk = self.g[i:end]                  # [Chunk]
            
            cross_term = torch.matmul(X_feat, Y_chunk.t())
            bias = g_chunk
            
            scores = cross_term + bias.unsqueeze(0) # [B, Chunk]
            
            chunk_max_scores, chunk_max_indices = torch.max(scores, dim=1)
            
            mask = chunk_max_scores > best_scores
            best_scores[mask] = chunk_max_scores[mask]
            best_indices[mask] = chunk_max_indices[mask] + i
            
        return best_indices.cpu().numpy()
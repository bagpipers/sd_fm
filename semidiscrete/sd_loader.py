import torch
from torch.utils.data import IterableDataset

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM用データセット。
    ユーザーの理想形:
    - ノイズ x_raw (Raw Space) を生成して保持。
    - PCAが必要な場合、pca_processorを使ってカップリング探索時のみ次元削減を行う。
    - PCA関係のパラメータ(行列など)は個別に持たない。
    """
    def __init__(self, 
                 original_dataset, 
                 potential_g, 
                 dataset_features, 
                 feature_dim, 
                 device, 
                 batch_size=64, 
                 pca_processor=None, # PCA専用関数/クラスを受け取る
                 chunk_size=10000):
        
        self.dataset = original_dataset
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dataset_features = dataset_features.to(device) 
        self.g = potential_g.to(device)
        
        self.pca_processor = pca_processor
        if self.pca_processor is not None:
            sample_img = self.dataset[0]["image"] # Tensor [C, H, W]
            self.noise_dim = sample_img.numel()
        else:
            self.noise_dim = feature_dim

    def __iter__(self):
        while True:
            X_raw = torch.randn(self.batch_size, self.noise_dim, device=self.device)
            if self.pca_processor is not None:
                X_feat = self.pca_processor.transform(X_raw)
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
                "noise": X_raw,                            # x_0 (Paired Noise - Raw: もとのx_rawを使う)
                "positive_prompt": batch_pos,
                "negative_prompt": batch_neg
            }

    def _get_indices_chunked(self, X_feat):
        """
        メモリ節約のため、データセット側(Y)をチャンク分割して最大スコアを探索する。
        X_feat: [B, D] (PCA済み)
        Y_chunk: [Chunk, D] (PCA済み)
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
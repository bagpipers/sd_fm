import torch
import numpy as np
import random
from torch.utils.data import IterableDataset

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM用データセット。
    パフォーマンス改善版:
    - 特徴量とポテンシャルをCPUに保持し、カップリング探索(MIPS)をCPUで行う。
    - これにより num_workers > 0 での並列データ読み込みが可能となり、学習速度が向上する。

    SD-FMの理論上、ノイズはゼロ平均のガウス分布であるべき。
    ノイズから画像の平均値を引いてしまうと、分布の中心がズレてしまい、
    Solverで学習したポテンシャル $g$ との対応関係が破綻。(平均引き算を回避するため、回転行列(components_)のみを保持)
    """



    def __init__(self, 
                 original_dataset, 
                 potential_g, 
                 dataset_features, 
                 feature_dim, 
                 device, 
                 batch_size=64, 
                 pca_processor=None, 
                 chunk_size=10000):
        
        self.dataset = original_dataset
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.dataset_features = dataset_features.cpu().float()
        self.g = potential_g.cpu().float()
        self.device = device 
        
        self.pca_processor = pca_processor
        self.pca_components = None
        if self.pca_processor is not None:
            self.pca_components = torch.from_numpy(self.pca_processor.pca.components_).float()

        if self.pca_processor is not None:
            try:
                sample_item = self.dataset[0]
                if "image" in sample_item:
                    sample_img = sample_item["image"]
                    self.noise_dim = sample_img.numel()
                else:
                    self.noise_dim = feature_dim # Fallback
            except Exception as e:
                print(f"Error determining noise dimension: {e}")
                raise e
        else:
            self.noise_dim = feature_dim

    def _init_worker_seed(self):
        """
        マルチプロセス(num_workers > 0)実行時に、各ワーカーが異なる乱数シードを持つように初期化する。
        これを呼ばないと、全ワーカーが同じ乱数系列（同じノイズ）を生成してしまう。

        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            base_seed = torch.initial_seed() % (2**32)
            worker_seed = base_seed + worker_info.id
            torch.manual_seed(worker_seed)
            np.random.seed(worker_seed % (2**32))
            random.seed(worker_seed % (2**32))

    def __iter__(self):
        """
        DataLoaderのワーカープロセス内で実行されるイテレータ。
        """
        self._init_worker_seed()
        while True:
            X_raw = torch.randn(self.batch_size, self.noise_dim)
            if self.pca_components is not None:
                X_feat = torch.matmul(X_raw, self.pca_components.t())
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
                batch_neg.append(item.get("negative_prompt", ""))
                
            yield {
                "image": torch.stack(batch_pixels),
                "noise": X_raw,
                "positive_prompt": batch_pos,
                "negative_prompt": batch_neg
            }

    def _get_indices_chunked(self, X_feat):
        """
        メモリ節約のため、データセット側(Y)をチャンク分割して最大スコアを探索する。
        全ての計算を CPU 上で行う。
        Score = x^T y + g (内積コストの場合)
        """
        N = self.dataset_features.shape[0]
        B = X_feat.shape[0]
        best_scores = torch.full((B,), float('-inf'), device='cpu')
        best_indices = torch.zeros((B,), dtype=torch.long, device='cpu')
        
        for i in range(0, N, self.chunk_size):
            end = min(i + self.chunk_size, N)
            
            Y_chunk = self.dataset_features[i:end]   # [Chunk, D] (CPU)
            g_chunk = self.g[i:end]                  # [Chunk] (CPU)
            cross_term = torch.matmul(X_feat, Y_chunk.t())
            scores = cross_term + g_chunk.unsqueeze(0) 
            
            chunk_max_scores, chunk_max_indices = torch.max(scores, dim=1)
            
            mask = chunk_max_scores > best_scores
            best_scores[mask] = chunk_max_scores[mask]
            best_indices[mask] = chunk_max_indices[mask] + i
            
        return best_indices.numpy()
    
import torch
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from torch.utils.data import IterableDataset, Dataset

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM (Semidiscrete Flow Matching) 用のデータセットクラス。

    役割:
    1. バッチごとに正規ガウスノイズ x_0 (画像と同じ次元) を生成する。
    2. ノイズ x_0 を特徴量空間 (PCA次元) に射影する。
    3. 学習済みのポテンシャル g を用いて、最適なターゲット画像 x_1 のインデックスを探索する (MIPS)。
    4. ペアとなる (x_0, x_1) を返す。

    Attributes:
        dataset (Dataset): 元の画像データセット。インデックスアクセス可能である必要がある。
        batch_size (int): バッチサイズ。
        device (torch.device): 計算に使用するデバイス (CPU推奨)。
        dataset_features (torch.Tensor): 全画像の特徴量行列。Shape: [N, Feature_Dim]
        potential_g (torch.Tensor): 学習済みの双対変数(ポテンシャル)。Shape: [N]
        pca_components (Optional[torch.Tensor]): PCAの基底ベクトル (回転行列)。Shape: [Feature_Dim, Image_Dim]
        image_dim (int): 画像データのフラット化された次元数 (C * H * W)。
        chunk_size (int): MIPS探索時のメモリ節約用チャンクサイズ。


    注意・注釈:
        - 特徴量とポテンシャルをCPUに保持し、カップリング探索(MIPS)をCPUで行う。
        - これにより num_workers > 0 での並列データ読み込みが可能となり、学習速度が向上する。

        SD-FMの理論上、ノイズはゼロ平均のガウス分布であるべき。
        ノイズから画像の平均値を引いてしまうと、分布の中心がズレてしまい、
        Solverで学習したポテンシャル $g$ との対応関係が破綻。(平均引き算を回避するため、回転行列(components_)のみを保持)    
    """

    def __init__(self, 
                 original_dataset: Dataset, 
                 dataset_features: torch.Tensor,
                 potential_g: torch.Tensor,
                 device: torch.device, 
                 batch_size: int = 64, 
                 pca_components: Optional[torch.Tensor] = None, 
                 chunk_size: int = 10000):
        """
        Args:
            original_dataset (Dataset): 画像データセット。__getitem__で {"image": ...} を返すこと。
            dataset_features (torch.Tensor): 全データの特徴量 (PCA済み)。CPU上のTensor推奨。
            potential_g (torch.Tensor): 最適化されたポテンシャル g。CPU上のTensor推奨。
            device (torch.device): 演算デバイス。通常は 'cpu' (データローダーの並列化のため)。
            batch_size (int): 1回のイテレーションで生成するペアの数。
            pca_components (Optional[torch.Tensor]): PCAを使用している場合、その成分 (components_)。
                高次元ノイズを特徴量空間へ射影するために使用する。Noneの場合は射影を行わない。
            chunk_size (int): 検索時のメモリ使用量を抑えるためのチャンクサイズ。
        """
        self.dataset = original_dataset
        self.dataset_features = dataset_features.float().to(device)
        self.potential_g = potential_g.float().to(device)
        self.device = device
        self.batch_size = batch_size
        self.pca_components = pca_components
        self.chunk_size = chunk_size

        if self.pca_components is not None:
            self.pca_components = self.pca_components.float().to(device)
        self.image_dim = self._determine_image_dim()
        if self.pca_components is not None:
            if self.pca_components.shape[1] != self.image_dim:
                raise ValueError(
                    f"Dimension mismatch: PCA components expect input dim {self.pca_components.shape[1]}, "
                    f"but dataset image dim is {self.image_dim}."
                )

    def _determine_image_dim(self) -> int:
        """データセットから画像次元 (C*H*W) を取得する。"""
        try:
            sample_item = self.dataset[0]
            if "image" not in sample_item:
                raise KeyError("Dataset item must contain 'image' key.")
            return sample_item["image"].numel()
        except Exception as e:
            raise RuntimeError(f"Failed to determine image dimension from dataset: {e}")

    def _init_worker_seed(self):
        """
        マルチプロセス(num_workers > 0)実行時に、各ワーカーが異なる乱数シードを持つように初期化する。
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
        DataLoaderのイテレータ。
        無限ループで (ノイズ, 画像) のペアを生成し続ける。
        """
        self._init_worker_seed()
        
        while True:
            noise_high_dim = self._generate_high_dim_noise()
            noise_feat = self._project_to_feature_space(noise_high_dim)
            indices = self._search_best_pairings(noise_feat)
            batch_data = self._fetch_batch_data(indices)
            
            yield {
                "image": batch_data["images"],            # x_1 (Target Data)
                "noise": noise_high_dim,                  # x_0 (Source Noise)
                "positive_prompt": batch_data["pos_prompts"],
                "negative_prompt": batch_data["neg_prompts"]
            }

    def _generate_high_dim_noise(self) -> torch.Tensor:
        """
        学習に使用する画像と同じ次元の正規ガウスノイズを生成する。
        
        Returns:
            torch.Tensor: Shape [Batch, Image_Dim]
        """
        return torch.randn(self.batch_size, self.image_dim, device=self.device)

    def _project_to_feature_space(self, noise: torch.Tensor) -> torch.Tensor:
        """
        高次元ノイズを、ペアリング探索用の特徴量空間に射影する。
        PCAが有効な場合は回転行列を適用し、無効な場合はそのまま返す。

        Args:
            noise (torch.Tensor): [Batch, Image_Dim]

        Returns:
            torch.Tensor: [Batch, Feature_Dim]
        """
        if self.pca_components is not None:
            return torch.matmul(noise, self.pca_components.t())
        else:
            return noise

    def _search_best_pairings(self, query_noise: torch.Tensor) -> np.ndarray:
        """
        Semidiscrete OT の理論に基づき、スコアが最大となるデータセットのインデックスを探索する。
        Score(x, y_j) = <x, y_j> + g_j
        メモリ節約のため、データセット側をチャンク分割して計算する。

        Args:
            query_noise (torch.Tensor): 探索クエリとなるノイズ特徴量。 [Batch, Feature_Dim]

        Returns:
            np.ndarray: 各クエリに対応する最適なデータセットインデックス。 [Batch]
        """

        num_data = self.dataset_features.shape[0]
        batch_size = query_noise.shape[0]
        best_scores = torch.full((batch_size,), float('-inf'), device=self.device)
        best_indices = torch.zeros((batch_size,), dtype=torch.long, device=self.device)

        for start_idx in range(0, num_data, self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, num_data)
            features_chunk = self.dataset_features[start_idx:end_idx]   # [Chunk, Feat_Dim]
            potential_chunk = self.potential_g[start_idx:end_idx]       # [Chunk]
            cross_term = torch.matmul(query_noise, features_chunk.t())
            scores = cross_term + potential_chunk.unsqueeze(0) 
            chunk_max_scores, chunk_arg_max = torch.max(scores, dim=1)
            update_mask = chunk_max_scores > best_scores
            best_scores[update_mask] = chunk_max_scores[update_mask]
            best_indices[update_mask] = chunk_arg_max[update_mask] + start_idx
            
        return best_indices.cpu().numpy()

    def _fetch_batch_data(self, indices: np.ndarray) -> Dict[str, List]:
        """
        決定されたインデックスリストに基づき、データセットから実際のデータ(画像、プロンプト)を取得する。

        Args:
            indices (np.ndarray): データセットのインデックス配列。

        Returns:
            Dict: 画像スタックとプロンプトのリストを含む辞書。
        """
        images = []
        pos_prompts = []
        neg_prompts = []

        for idx in indices:
            item = self.dataset[int(idx)]
            images.append(item["image"]) 
            pos_prompts.append(str(item["positive_prompt"]))
            neg_prompts.append(str(item.get("negative_prompt", "")))

        return {
            "images": torch.stack(images),
            "pos_prompts": pos_prompts,
            "neg_prompts": neg_prompts
        }
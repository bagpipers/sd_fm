import os
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from typing import Optional

from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset

class OnlinePCAProcessor:
    """Incremental PCA (IPCA) を使用し、メモリ爆発を防ぎながら学習・変換を行うクラス。"""
    def __init__(self, n_components: int, device: str):
        self.n_components = n_components
        self.device = device
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False

    def fit_incremental(self, dataloader: DataLoader):
        """データローダーからデータを取得し、IPCAで部分学習を行う。"""
        print(f"Fitting PCA incrementally (dim={self.n_components})...")
        for batch in tqdm(dataloader, desc="PCA Fitting"):
            imgs = batch["image"]
            x_flat = imgs.view(imgs.shape[0], -1).cpu().numpy()
            self.pca.partial_fit(x_flat)
        self.is_fitted = True
        print("PCA fitting complete.")

    def transform(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        TensorデータをPCA空間に射影する。
        注意: 画像データ生成におけるSD-FMの理論的整合性のため、平均引きを行わず回転のみを適用する。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_cpu = x_tensor.detach().cpu().numpy()
        x_pca = np.dot(x_cpu, self.pca.components_.T)
        
        return torch.from_numpy(x_pca).to(self.device).float()

    def get_components_tensor(self) -> torch.Tensor:
        """PCAの基底ベクトル(回転行列)をTensorとして返す。"""
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return torch.from_numpy(self.pca.components_).float().to(self.device)

    def save(self, path: str):
        joblib.dump(self.pca, path)
        print(f"PCA model saved to {path}")

    def load(self, path: str):
        self.pca = joblib.load(path)
        self.is_fitted = True
        print(f"PCA model loaded from {path}")


class SD_Manager:
    """
    SD-FMの準備フェーズを統括し、データローダーを構築するマネージャークラス。
    特徴量抽出、PCA学習、ポテンシャル学習(Phase 1)、データセット構築を行う。

    注意・注釈
        SD-FMの理論上、ノイズはゼロ平均のガウス分布であるべき。
        ノイズから画像の平均値を引いてしまうと、分布の中心がズレてしまい、
        Solverで学習したポテンシャル $g$ との対応関係が破綻。(平均引き算を回避するため、回転行列(components_)のみを保持)  
    """
    def __init__(self, config: dict, device: str):
        self.config = config
        self.sd_config = config.get('sd_config', {})
        self.device = device
        
        self.save_dir = config['training'].get('save_dir', 'outputs')
        os.makedirs(self.save_dir, exist_ok=True)
        self.potential_path = os.path.join(self.save_dir, "sd_potential.pt")
        self.pca_model_path = os.path.join(self.save_dir, "pca_model.joblib")
        self.features_cache_path = os.path.join(self.save_dir, "cached_features.pt")
        self.raw_dim = config['data']['height'] * config['data']['width'] * config['data']['channels']
        self.use_pca = self.sd_config.get('use_pca', False)
        self.feature_dim = self.sd_config['pca_dim'] if self.use_pca else self.raw_dim
        
        if self.use_pca:
            self.pca_processor = OnlinePCAProcessor(self.feature_dim, device)
        else:
            self.pca_processor = None

    def prepare_dataloader(self, raw_dataset: Dataset) -> DataLoader:
        """
        生データセットを受け取り、SD-FM用のペアリングデータローダーを返す。
        必要に応じて、特徴量抽出・PCA学習・ポテンシャル学習を実行またはキャッシュからロードする。
        """
        print("\n=== [SD-FM Manager] Preparing Data & Potential... ===")
        features_tensor = self._prepare_features(raw_dataset)
        potential_g = self._prepare_potential(features_tensor, len(raw_dataset))
        pca_components_tensor = None
        if self.use_pca and self.pca_processor is not None:
            pca_components_tensor = self.pca_processor.get_components_tensor()
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            dataset_features=features_tensor,
            potential_g=potential_g,
            device=torch.device('cpu'), # データローダー内はCPUで計算し、マルチプロセス化する
            batch_size=self.config['training']['batch_size'],
            pca_components=pca_components_tensor.cpu() if pca_components_tensor is not None else None,
            chunk_size=self.sd_config.get('pairing_chunk_size', 10000)
        )
        num_workers = self.config['training'].get('num_workers', 0)
        print(f"SD-FM Dataloader created. (num_workers={num_workers})")
        
        return DataLoader(
            sd_dataset, 
            batch_size=None,   
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True
        )

    def _prepare_potential(self, features_tensor: torch.Tensor, dataset_size: int) -> torch.Tensor:
        """ポテンシャル g をロードするか、なければ学習する。"""
        if os.path.exists(self.potential_path):
            print(f"Loading existing potential from {self.potential_path}")
            return torch.load(self.potential_path, map_location=self.device)
        
        print("Potential not found. Starting SD-OT training (Phase 1)...")
        solver = SemidiscreteOT_Solver(
            dataset_size=dataset_size,
            feature_dim=self.feature_dim,
            device=self.device,
            batch_size_noise=self.sd_config.get('potential_batch_size', 1024),
            lr=self.sd_config.get('potential_lr', 0.1)
        )
        g_ema = solver.train_loop(
            flattened_dataset_tensor=features_tensor,
            num_iterations=self.sd_config.get('potential_steps', 20000)
        )
        torch.save(g_ema, self.potential_path)
        print(f"Potential saved to {self.potential_path}")
        return g_ema

    def _prepare_features(self, dataset: Dataset) -> torch.Tensor:
        """
        データセットから特徴量を抽出する。
        PCA使用時は学習と変換を行う。キャッシュがあればロードする。
        """
        if os.path.exists(self.features_cache_path):
            if self.use_pca:
                if os.path.exists(self.pca_model_path):
                    print(f"Loading cached features and PCA model...")
                    self.pca_processor.load(self.pca_model_path)
                    features = torch.load(self.features_cache_path, map_location="cpu")
                    features.share_memory_() # マルチプロセス共有用
                    return features
            else:
                print(f"Loading cached raw features...")
                features = torch.load(self.features_cache_path, map_location="cpu")
                features.share_memory_()
                return features
        print("Extracting features from dataset...")
        temp_loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
        
        if self.use_pca:
            if not os.path.exists(self.pca_model_path):
                self.pca_processor.fit_incremental(temp_loader)
                self.pca_processor.save(self.pca_model_path)
            else:
                self.pca_processor.load(self.pca_model_path)
            all_features = []
            print("Transforming all data with PCA...")
            for batch in tqdm(temp_loader, desc="PCA Transform"):
                imgs = batch["image"]
                flat = imgs.view(imgs.shape[0], -1) 
                feat = self.pca_processor.transform(flat) 
                all_features.append(feat.cpu())
                
            full_tensor = torch.cat(all_features, dim=0) # [N, PCA_Dim]
        else:
            print("Flattening raw features (No PCA)...")
            all_tensors = []
            for batch in tqdm(temp_loader, desc="Flattening"):
                imgs = batch["image"]
                flat = imgs.view(imgs.shape[0], -1)
                all_tensors.append(flat.cpu()) 
            full_tensor = torch.cat(all_tensors, dim=0)
        print(f"Saving features cache to {self.features_cache_path}")
        torch.save(full_tensor, self.features_cache_path)
        full_tensor.share_memory_()
        
        return full_tensor
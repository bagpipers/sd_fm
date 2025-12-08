import os
import torch
import joblib
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA

from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset

class OnlinePCAProcessor:
    """
    Incremental PCA (IPCA) を使用し、メモリ爆発を防ぎながら学習・変換を行う。
    sd_solver.py や sd_loader.py からも利用可能。
    """
    def __init__(self, n_components, device):
        self.n_components = n_components
        self.device = device
        
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False

    def fit_incremental(self, dataloader):
        """
        データローダーからバッチごとにデータを取得し、IPCAで学習する。
        全データをメモリに展開しないため、省メモリ。
        """
        print(f"Fitting PCA incrementally (dim={self.n_components})...")
        for batch in tqdm(dataloader, desc="PCA Fitting"):
            
            imgs = batch["pixel_values"]
            x_flat = imgs.view(imgs.shape[0], -1).cpu().numpy()
            self.pca.partial_fit(x_flat)
        self.is_fitted = True
        print("PCA fitting complete.")

    def transform(self, x_tensor):
        """
        Tensor (on device) -> PCA Transform -> Tensor (on device)
        x_tensor: [B, Raw_Dim]
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        x_cpu = x_tensor.detach().cpu().numpy()
        x_pca = self.pca.transform(x_cpu)
        return torch.from_numpy(x_pca).to(self.device).float()

    def save(self, path):
        joblib.dump(self.pca, path)
        print(f"PCA model saved to {path}")

    def load(self, path):
        self.pca = joblib.load(path)
        self.is_fitted = True
        print(f"PCA model loaded from {path}")


class SD_Manager:
    """
    SD-FMの準備とデータローダー構築を行うマネージャー。
    """
    def __init__(self, config, device):
        self.config = config
        self.sd_config = config['sd_config']
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

    def prepare_dataloader(self, raw_dataset):
        print("\n=== [SD-FM Manager] Preparing Data & Potential... ===")
        features_tensor = self._prepare_features(raw_dataset)
        if os.path.exists(self.potential_path):
            print(f"Loading existing potential from {self.potential_path}")
            g_ema = torch.load(self.potential_path, map_location=self.device)
        else:
            print("Potential not found. Starting SD-OT training (Phase 1)...")
            solver = SemidiscreteOT_Solver(
                dataset_size=len(raw_dataset),
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
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            potential_g=g_ema,
            dataset_features=features_tensor,
            feature_dim=self.feature_dim,
            device=self.device,
            batch_size=self.config['training']['batch_size'],
            pca_processor=self.pca_processor, # 行列ではなくプロセッサを渡す
            chunk_size=self.sd_config.get('pairing_chunk_size', 10000)
        )
        
        return DataLoader(sd_dataset, batch_size=None, num_workers=0)

    def _prepare_features(self, dataset):
        """
        IPCAを用いてメモリ爆発を回避しながら全画像の特徴量を抽出する。
        """
        if os.path.exists(self.features_cache_path):
            if self.use_pca:
                if os.path.exists(self.pca_model_path):
                    print(f"Loading cached features and PCA model...")
                    self.pca_processor.load(self.pca_model_path)
                    return torch.load(self.features_cache_path, map_location="cpu")
            else:
                print(f"Loading cached raw features...")
                return torch.load(self.features_cache_path, map_location="cpu")

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
                imgs = batch["pixel_values"]
                flat = imgs.view(imgs.shape[0], -1) # [B, Raw]
                feat = self.pca_processor.transform(flat) 
                all_features.append(feat.cpu())
                
            full_tensor = torch.cat(all_features, dim=0) # [N, PCA_Dim]
        else:
            print("Flattening raw features (No PCA)...")
            all_tensors = []
            for batch in tqdm(temp_loader, desc="Flattening"):
                imgs = batch["pixel_values"]
                flat = imgs.view(imgs.shape[0], -1)
                all_tensors.append(flat.cpu()) # CPUへ退避
            full_tensor = torch.cat(all_tensors, dim=0)
        print(f"Saving features cache to {self.features_cache_path}")
        torch.save(full_tensor, self.features_cache_path)
        
        return full_tensor
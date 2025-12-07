import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset

class SD_Manager:
    """
    SD-FMの準備とデータローダー構築を行うマネージャー。
    - PCAの計算と適用
    - ポテンシャル(g)の学習またはロード
    - SD-Loaderの構築
    """
    def __init__(self, config, device):
        self.config = config
        self.sd_config = config['sd_config']
        self.device = device
        self.save_dir = config['training'].get('save_dir', 'outputs')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.potential_path = os.path.join(self.save_dir, "sd_potential.pt")
        self.pca_matrix_path = os.path.join(self.save_dir, "pca_matrix.pt")
        self.pca_mean_path = os.path.join(self.save_dir, "pca_mean.pt")
        self.raw_dim = config['data']['height'] * config['data']['width'] * config['data']['channels']
        self.use_pca = self.sd_config.get('use_pca', False)
        self.feature_dim = self.sd_config['pca_dim'] if self.use_pca else self.raw_dim

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
        pca_proj = None
        pca_mean = None
        if self.use_pca:
            pca_proj = torch.load(self.pca_matrix_path, map_location=self.device)
            pca_mean = torch.load(self.pca_mean_path, map_location=self.device)
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            potential_g=g_ema,
            dataset_features=features_tensor,
            feature_dim=self.feature_dim,
            device=self.device,
            batch_size=self.config['training']['batch_size'],
            pca_proj=pca_proj,
            pca_mean=pca_mean,
            chunk_size=self.sd_config.get('pairing_chunk_size', 10000)
        )
        
        return DataLoader(sd_dataset, batch_size=None, num_workers=0)

    def _prepare_features(self, dataset):
        """
        全画像を読み込み、Flattenし、必要ならPCAを学習・適用して返す。
        """
        features_cache_path = os.path.join(self.save_dir, "cached_features.pt")
        if os.path.exists(features_cache_path) and self.use_pca and os.path.exists(self.pca_matrix_path):
            print(f"Loading cached features from {features_cache_path}")
            return torch.load(features_cache_path, map_location="cpu") # 一旦CPUへ

        print("Flattening dataset for feature extraction...")
        temp_loader = DataLoader(dataset, batch_size=256, num_workers=8, shuffle=False)
        all_tensors = []
        
        for batch in tqdm(temp_loader, desc="Loading Data"):
            imgs = batch["pixel_values"] # T2IDatasetは pixel_values を返す
            flat = imgs.view(imgs.shape[0], -1)
            all_tensors.append(flat)
        
        full_tensor = torch.cat(all_tensors, dim=0) # [N, Raw_Dim]
        
        if self.use_pca:
            print(f"Computing PCA (dim={self.feature_dim})... This may take a while.")
            device_calc = self.device if full_tensor.shape[0] < 50000 else "cpu" 
            X = full_tensor.to(device_calc)
            mean = torch.mean(X, dim=0)
            X_centered = X - mean
            U, S, V = torch.pca_lowrank(X_centered, q=self.feature_dim, center=False, niter=2)
            features = torch.matmul(X_centered, V)
            torch.save(V.to("cpu"), self.pca_matrix_path)
            torch.save(mean.to("cpu"), self.pca_mean_path)
            
            print(f"PCA done. Features shape: {features.shape}")
            features = features.cpu() # メモリ節約のため一旦CPU
            torch.save(features, features_cache_path)
            return features
            
        else:
            print("PCA disabled. Using raw features.")
            return full_tensor
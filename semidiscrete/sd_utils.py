import os
import torch
import joblib
import numpy as np
import json
import shutil
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from typing import Optional

from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset

class OnlinePCAProcessor:
    """
    Incremental PCA (IPCA) を使用し、メモリ爆発を防ぎながら学習・変換を行うクラス。
    """
    def __init__(self, n_components: int, device: str):
        self.n_components = n_components
        self.device = device
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False

    def fit_incremental(self, dataset: Dataset, batch_size: int, num_workers: int, checkpoint_path: Optional[str] = None, save_interval: int = 5000):
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming PCA fitting from {checkpoint_path}...")
            try:
                self.pca = joblib.load(checkpoint_path)
            except Exception as e:
                print(f" -> Failed to load checkpoint: {e}. Starting from scratch.")
        samples_seen = self.pca.n_samples_seen_ if hasattr(self.pca, 'n_samples_seen_') else 0
        total_samples = len(dataset)
        
        if samples_seen >= total_samples:
            print(f"PCA fitting already completed (seen {samples_seen}/{total_samples} samples).")
            self.is_fitted = True
            return

        print(f"Fitting PCA incrementally... Resuming from sample {samples_seen}")
        
        remaining_indices = range(samples_seen, total_samples)
        subset = Subset(dataset, remaining_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        buffer_list = []
        buffer_size = 0
        min_batch_size = self.n_components 
        samples_since_save = 0

        for batch in tqdm(dataloader, desc="PCA Fitting"):
            imgs = batch["image"]
            current_batch_size = imgs.shape[0]
            
            x_flat = imgs.view(current_batch_size, -1).numpy()
            
            buffer_list.append(x_flat)
            buffer_size += current_batch_size
            samples_since_save += current_batch_size
            
            if buffer_size >= min_batch_size:
                X_batch = np.concatenate(buffer_list, axis=0)
                self.pca.partial_fit(X_batch)
                buffer_list = []
                buffer_size = 0
                
                if checkpoint_path and samples_since_save >= save_interval:
                    self._save_checkpoint(checkpoint_path)
                    samples_since_save = 0

        if buffer_size > 0:
            X_batch = np.concatenate(buffer_list, axis=0)
            if buffer_size >= self.n_components or self.pca.n_samples_seen_ > 0:
                 self.pca.partial_fit(X_batch)
        
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path)

        self.is_fitted = True
        print("PCA fitting complete.")

    def _save_checkpoint(self, path):
        temp_path = path + ".tmp"
        joblib.dump(self.pca, temp_path)
        if os.path.exists(temp_path):
            os.replace(temp_path, path)

    def transform(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        学習・推論時用: Tensorを受け取り、GPU Tensorを返す。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_cpu = x_tensor.detach().cpu().numpy()
        x_pca = np.dot(x_cpu, self.pca.components_.T)
        
        return torch.from_numpy(x_pca).to(self.device).float()

    def transform_numpy(self, x_numpy: np.ndarray) -> np.ndarray:
        """
        【追加】データ前処理用: NumPyを受け取り、NumPyを返す。
        GPU転送を行わないため、ディスク保存処理などで高速。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return np.dot(x_numpy, self.pca.components_.T)

    def get_components_tensor(self) -> torch.Tensor:
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
            device=torch.device('cpu'),
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
        全画像の特徴量を抽出し、キャッシュする。
        【最適化済み】NumPy (mmap) を活用し、無駄なGPU転送を回避。
        """
        if os.path.exists(self.features_cache_path):
            if self.use_pca and not self.pca_processor.is_fitted:
                if os.path.exists(self.pca_model_path):
                    print("Loading PCA model for cached features...")
                    self.pca_processor.load(self.pca_model_path)
            
            print(f"Loading cached features from {self.features_cache_path}")
            features = torch.load(self.features_cache_path, map_location="cpu")
            features.share_memory_()
            return features

        print("Extracting features from dataset...")
        
        if self.use_pca:
            if not os.path.exists(self.pca_model_path):
                fit_batch_size = 256 
                self.pca_processor.fit_incremental(
                    dataset, 
                    batch_size=fit_batch_size, 
                    num_workers=4, 
                    checkpoint_path=os.path.join(self.save_dir, "pca_checkpoint_fit.joblib")
                )
                self.pca_processor.save(self.pca_model_path)
            else:
                self.pca_processor.load(self.pca_model_path)

        total_samples = len(dataset)
        temp_mmap_path = os.path.join(self.save_dir, "features_temp.mmap")
        progress_path = os.path.join(self.save_dir, "features_progress.json")
        mode = 'r+' if os.path.exists(temp_mmap_path) else 'w+'
        mmap_features = np.memmap(temp_mmap_path, dtype='float32', mode=mode, shape=(total_samples, self.feature_dim))
        
        processed_count = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    data = json.load(f)
                    processed_count = data.get("processed_count", 0)
                print(f"Resuming feature extraction from sample {processed_count}...")
            except:
                print("Failed to load progress file. Starting from 0.")
        
        if processed_count < total_samples:
            subset = Subset(dataset, range(processed_count, total_samples))
            transform_batch_size = 256
            loader = DataLoader(subset, batch_size=transform_batch_size, num_workers=4, shuffle=False)
            
            current_idx = processed_count
            
            print(f"Transforming samples {processed_count} to {total_samples}...")
            for batch in tqdm(loader, desc="Extracting Features"):
                imgs = batch["image"]
                batch_len = imgs.shape[0]
                x_numpy = imgs.view(batch_len, -1).numpy()
                
                if self.use_pca:
                    feat = self.pca_processor.transform_numpy(x_numpy)
                else:
                    feat = x_numpy
                mmap_features[current_idx : current_idx + batch_len] = feat
                current_idx += batch_len
                with open(progress_path, 'w') as f:
                    json.dump({"processed_count": current_idx}, f)
            
            mmap_features.flush()
        
        print("Feature extraction complete. Converting to Torch Tensor...")
        full_tensor = torch.from_numpy(mmap_features)
        torch.save(full_tensor, self.features_cache_path)
        
        print(f"Features saved to {self.features_cache_path}")
        
        del mmap_features 
        if os.path.exists(temp_mmap_path):
            os.remove(temp_mmap_path)
        if os.path.exists(progress_path):
            os.remove(progress_path)
            
        final_features = torch.load(self.features_cache_path, map_location="cpu")
        final_features.share_memory_()
        
        return final_features
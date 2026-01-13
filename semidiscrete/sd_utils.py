import os
import torch
import numpy as np
import json
import shutil
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from typing import Optional
from .incremental_pca import IncrementalPCA
from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset


class OnlinePCAProcessor:
    """
    PyTorchベースの Incremental PCA を使用し、GPU/CPU切り替え可能なクラス。
    大規模データセットに対して、バッチごとにPCAを適用・更新します。
    """
    def __init__(self, n_components: int, device: str):
        """
        Args:
            n_components (int): 圧縮後の次元数 (K)。
            device (str): 計算に使用するデバイス ('cuda' or 'cpu')。
        """
        self.n_components = n_components
        self.device = torch.device(device)
        self.pca = IncrementalPCA(n_components=n_components, copy=False)
        self.is_fitted = False
        
        print(f"OnlinePCAProcessor initialized on {self.device}")

    def fit_incremental(self, dataset: Dataset, batch_size: int, num_workers: int, checkpoint_path: Optional[str] = None, save_interval: int = 5000):
        """
        PyTorch IncrementalPCA を用いて学習を行います。
        
        重要: IPCAの仕様上、batch_size (B) は n_components (K) 以上である必要があります。
        """
        if batch_size < self.n_components:
            raise ValueError(
                f"\n[Config Error] PCA fit batch_size ({batch_size}) must be >= n_components ({self.n_components}).\n"
                f"Incremental PCA requires the batch size to be larger than the number of components.\n"
                f"Please update 'pca_fit_batch_size' (or training batch_size) in 'configs/config_sd.yaml' to be at least {self.n_components}.\n"
            )
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming PCA fitting from {checkpoint_path}...")
            try:
                self.load(checkpoint_path)
            except Exception as e:
                print(f" -> Failed to load checkpoint: {e}. Starting from scratch.")

        samples_seen = 0
        if hasattr(self.pca, 'n_samples_seen_') and self.pca.n_samples_seen_ is not None:
             samples_seen = int(self.pca.n_samples_seen_.item())

        total_samples = len(dataset)
        
        if samples_seen >= total_samples:
            print(f"PCA fitting already completed (seen {samples_seen}/{total_samples} samples).")
            self.is_fitted = True
            return

        print(f"Fitting PCA incrementally on {self.device}... Resuming from sample {samples_seen}")
        
        remaining_indices = range(samples_seen, total_samples)
        if len(remaining_indices) == 0:
             self.is_fitted = True
             return

        subset = Subset(dataset, remaining_indices)
        use_pin_memory = (self.device.type == 'cuda')
        dataloader = DataLoader(
            subset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False,
            pin_memory=use_pin_memory 
        )

        samples_since_save = 0
        self._move_pca_to_device()

        for batch in tqdm(dataloader, desc=f"PCA Fitting ({self.device})"):
            imgs = batch["image"]
            current_batch_size = imgs.shape[0]
            if current_batch_size < self.n_components:
                print(f"\nWarning: Skipping a batch of size {current_batch_size} because it is smaller than n_components ({self.n_components}).")
                continue
            x_flat = imgs.view(current_batch_size, -1).to(self.device, dtype=torch.float32)
            self.pca.partial_fit(x_flat)
            
            samples_since_save += current_batch_size
            if checkpoint_path and samples_since_save >= save_interval:
                self._save_checkpoint(checkpoint_path)
                samples_since_save = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
            except OSError:
                pass

        self.is_fitted = True
        print("PCA fitting complete.")

    def _move_pca_to_device(self):
        """PCAの内部状態を指定デバイスに転送するヘルパー"""
        if not hasattr(self.pca, 'components_'): return

        if self.pca.components_ is not None: self.pca.components_ = self.pca.components_.to(self.device)
        if self.pca.mean_ is not None: self.pca.mean_ = self.pca.mean_.to(self.device)
        if self.pca.var_ is not None: self.pca.var_ = self.pca.var_.to(self.device)
        if self.pca.singular_values_ is not None: self.pca.singular_values_ = self.pca.singular_values_.to(self.device)
        if self.pca.n_samples_seen_ is not None: self.pca.n_samples_seen_ = self.pca.n_samples_seen_.to(self.device)

    def _save_checkpoint(self, path):
        """モデル全体を保存 (アトミック書き込み)"""
        temp_path = path + ".tmp"
        torch.save(self.pca, temp_path)
        if os.path.exists(temp_path):
            os.replace(temp_path, path)

    def transform(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """入力テンソル (CPU or GPU) を受け取り、PCA変換後の Tensor (GPU) を返す。"""
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_in = x_tensor.view(x_tensor.shape[0], -1).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            x_pca = self.pca.transform(x_in)
        
        return x_pca 

    def transform_numpy(self, x_numpy: np.ndarray) -> np.ndarray:
        """NumPy配列 (CPU) -> PCA変換 (GPU) -> NumPy配列 (CPU)"""
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_tensor = torch.from_numpy(x_numpy).to(self.device, dtype=torch.float32)
        with torch.no_grad():
            x_pca_tensor = self.pca.transform(x_tensor)
        
        return x_pca_tensor.detach().cpu().numpy()

    def get_components_tensor(self) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return self.pca.components_.to(self.device)

    def save(self, path: str):
        torch.save(self.pca, path)
        print(f"PCA model saved to {path}")

    def load(self, path: str):
        self.pca = torch.load(path, map_location=self.device)
        self.is_fitted = True
        self._move_pca_to_device() # 念のため再配置
        print(f"PCA model loaded from {path}")


class SD_Manager:
    """
    SD-FM の学習に必要な前処理を一括管理するクラス。
    """
    def __init__(self, config: dict, device: str):
        self.config = config
        self.sd_config = config.get('sd_config', {})
        self.device = torch.device(device) 
        
        self.save_dir = config['training'].get('save_dir', 'outputs')
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.potential_path = os.path.join(self.save_dir, "sd_potential.pt")
        self.potential_ckpt_path = os.path.join(self.save_dir, "sd_potential_checkpoint.pt")
        self.pca_model_path = os.path.join(self.save_dir, "pca_model.pt") 
        self.features_mmap_path = os.path.join(self.save_dir, "features.mmap")
        self.features_done_flag = os.path.join(self.save_dir, "features_done.flag")
        
        self.raw_dim = config['data']['height'] * config['data']['width'] * config['data']['channels']
        self.use_pca = self.sd_config.get('use_pca', False)
        self.feature_dim = self.sd_config['pca_dim'] if self.use_pca else self.raw_dim
        self.pca_device = self.sd_config.get('pca_device', 'cuda')
        
        if self.use_pca:
            print(f"SD_Manager: Initializing OnlinePCAProcessor on device '{self.pca_device}'")
            self.pca_processor = OnlinePCAProcessor(self.feature_dim, self.pca_device)
        else:
            self.pca_processor = None

    def prepare_dataloader(self, raw_dataset: Dataset) -> DataLoader:
        print("\n=== [SD-FM Manager] Preparing Data & Potential... ===")
        features_tensor = self._prepare_features(raw_dataset)
        potential_g = self._prepare_potential(features_tensor, len(raw_dataset))
        
        pca_components_tensor = None
        if self.use_pca and self.pca_processor is not None:
            pca_components_tensor = self.pca_processor.get_components_tensor().cpu()
            
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            dataset_features=features_tensor, 
            potential_g=potential_g.cpu(),    
            device=torch.device('cpu'),       
            batch_size=self.config['training']['batch_size'],
            pca_components=pca_components_tensor, 
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
            print(f"Loading finished potential from {self.potential_path}")
            try:
                data = torch.load(self.potential_path, map_location=self.device)
                if isinstance(data, dict) and 'g_ema' in data: return data['g_ema']
                return data
            except:
                return torch.load(self.potential_path, map_location=self.device)
        
        print("Potential not found (or incomplete). Starting/Resuming SD-OT training (Phase 1)...")
        solver = SemidiscreteOT_Solver(
            dataset_size=dataset_size,
            feature_dim=self.feature_dim,
            device=self.device, 
            batch_size_noise=self.sd_config.get('potential_batch_size', 1024),
            lr=self.sd_config.get('potential_lr', 0.1)
        )
        g_ema = solver.train_loop(
            flattened_dataset_tensor=features_tensor, 
            num_iterations=self.sd_config.get('potential_steps', 20000),
            chunk_size=self.sd_config.get('pairing_chunk_size', 10000),
            checkpoint_path=self.potential_ckpt_path, 
            save_interval=5000
        )
        torch.save(g_ema, self.potential_path)
        print(f"Potential training finished and saved to {self.potential_path}")
        if os.path.exists(self.potential_ckpt_path): os.remove(self.potential_ckpt_path)
        return g_ema

    def _prepare_features(self, dataset: Dataset) -> torch.Tensor:
        total_samples = len(dataset)
        
        if os.path.exists(self.features_mmap_path) and os.path.exists(self.features_done_flag):
            if self.use_pca and not self.pca_processor.is_fitted:
                if os.path.exists(self.pca_model_path):
                    print("Loading PCA model for cached features...")
                    self.pca_processor.load(self.pca_model_path)
            print(f"Loading cached features via mmap from {self.features_mmap_path}")
            mmap_features = np.memmap(self.features_mmap_path, dtype='float32', mode='r', shape=(total_samples, self.feature_dim))
            return torch.from_numpy(mmap_features)

        print("Extracting features from dataset...")
        if self.use_pca:
            if not os.path.exists(self.pca_model_path):
                fit_batch_size = self.sd_config.get('pca_fit_batch_size', 4096)
                
                print(f"Starting PCA incremental fit with batch size {fit_batch_size}...")
                self.pca_processor.fit_incremental(
                    dataset, 
                    batch_size=fit_batch_size, 
                    num_workers=4,
                    checkpoint_path=os.path.join(self.save_dir, "pca_checkpoint_fit.pt")
                )
                self.pca_processor.save(self.pca_model_path)
            else:
                self.pca_processor.load(self.pca_model_path)
        
        progress_path = os.path.join(self.save_dir, "features_progress.json")
        processed_count = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    data = json.load(f)
                    processed_count = data.get("processed_count", 0)
                print(f"Resuming feature extraction from sample {processed_count}...")
            except: pass
        
        mode = 'r+' if processed_count > 0 and os.path.exists(self.features_mmap_path) else 'w+'
        if mode == 'w+': processed_count = 0 

        mmap_features = np.memmap(self.features_mmap_path, dtype='float32', mode=mode, shape=(total_samples, self.feature_dim))
        
        if processed_count < total_samples:
            subset = Subset(dataset, range(processed_count, total_samples))
            loader = DataLoader(subset, batch_size=256, num_workers=4, shuffle=False) # 変換時はバッチサイズ小さくてもOK
            
            current_idx = processed_count
            samples_since_flush = 0
            FLUSH_INTERVAL = 10000 

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
                samples_since_flush += batch_len
                if samples_since_flush >= FLUSH_INTERVAL:
                    mmap_features.flush()
                    samples_since_flush = 0
                    with open(progress_path, 'w') as f: json.dump({"processed_count": current_idx}, f)

            mmap_features.flush()
        
        print("Feature extraction complete.")
        with open(self.features_done_flag, 'w') as f: f.write("done")
        if os.path.exists(progress_path): os.remove(progress_path)
        del mmap_features
        mmap_features = np.memmap(self.features_mmap_path, dtype='float32', mode='r', shape=(total_samples, self.feature_dim))
        
        return torch.from_numpy(mmap_features)
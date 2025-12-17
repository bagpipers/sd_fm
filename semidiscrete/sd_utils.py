import os
import torch
import joblib
import numpy as np
import shutil
import tempfile
import re
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA
from typing import Optional, List

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

    def fit_incremental(self, dataset: Dataset, batch_size: int, num_workers: int, checkpoint_path: Optional[str] = None, save_interval: int = 100):
        """
        DataLoaderを引数ではなく、Datasetを受け取り、内部でSubsetを使って高速にレジュームするように変更
        """
        start_batch_idx = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Resuming PCA fitting from {checkpoint_path}...")
            try:
                checkpoint = joblib.load(checkpoint_path)
                self.pca = checkpoint['pca']
                start_batch_idx = checkpoint['batch_idx']
                print(f" -> Resuming from batch index {start_batch_idx}")
            except Exception as e:
                print(f" -> Failed to load checkpoint: {e}. Starting from scratch.")
        
        total_samples = len(dataset)
        start_sample_idx = start_batch_idx * batch_size
        
        if start_sample_idx >= total_samples:
            print("PCA fitting already completed according to checkpoint.")
            self.is_fitted = True
            return
            
        remaining_indices = range(start_sample_idx, total_samples)
        subset = Subset(dataset, remaining_indices)
        dataloader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        print(f"Fitting PCA incrementally (dim={self.n_components})...")
        buffer_list = []
        buffer_size = 0
        min_batch_size = self.n_components 
        next_save_target = start_batch_idx + save_interval

        for i_local, batch in enumerate(tqdm(dataloader, desc="PCA Fitting")):
            current_batch_idx = start_batch_idx + i_local
            
            imgs = batch["image"]
            x_flat = imgs.view(imgs.shape[0], -1).cpu().numpy()
            
            buffer_list.append(x_flat)
            buffer_size += x_flat.shape[0]
            if buffer_size >= min_batch_size:
                X_batch = np.concatenate(buffer_list, axis=0)
                self.pca.partial_fit(X_batch)
                buffer_list = []
                buffer_size = 0
                if checkpoint_path and (current_batch_idx + 1) >= next_save_target:
                    self._save_checkpoint(checkpoint_path, current_batch_idx + 1)
                    next_save_target = (current_batch_idx + 1) + save_interval
        if buffer_size > 0:
            X_batch = np.concatenate(buffer_list, axis=0)
            if buffer_size >= self.n_components or self.pca.n_samples_seen_ > 0:
                 self.pca.partial_fit(X_batch)
        if checkpoint_path:
            self._save_checkpoint(checkpoint_path, start_batch_idx + len(dataloader))

        self.is_fitted = True
        print("PCA fitting complete.")

    def _save_checkpoint(self, path, batch_idx):
        """アトミック書き込みでチェックポイントを保存"""
        checkpoint = {
            'pca': self.pca,
            'batch_idx': batch_idx
        }
        dir_name = os.path.dirname(path)
        with tempfile.NamedTemporaryFile(delete=False, dir=dir_name, suffix='.tmp') as tmp_file:
            joblib.dump(checkpoint, tmp_file.name)
            tmp_name = tmp_file.name
        
        if tmp_name:
            os.replace(tmp_name, path)

    def transform(self, x_tensor: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_cpu = x_tensor.detach().cpu().numpy()
        x_pca = np.dot(x_cpu, self.pca.components_.T)
        
        return torch.from_numpy(x_pca).to(self.device).float()

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
        if os.path.exists(self.features_cache_path):
            if self.use_pca:
                if os.path.exists(self.pca_model_path):
                    print(f"Loading cached features and PCA model...")
                    self.pca_processor.load(self.pca_model_path)
                    features = torch.load(self.features_cache_path, map_location="cpu")
                    features.share_memory_() 
                    return features
            else:
                print(f"Loading cached raw features...")
                features = torch.load(self.features_cache_path, map_location="cpu")
                features.share_memory_()
                return features

        print("Extracting features from dataset...")
        pca_ckpt_path = os.path.join(self.save_dir, "pca_fit_checkpoint.joblib")
        feat_chunks_dir = os.path.join(self.save_dir, "feature_chunks_temp")
        os.makedirs(feat_chunks_dir, exist_ok=True)
        
        batch_size = 256 # 固定 (※注意: レジュームの整合性のため、この値は変更しないでください)
        num_workers = 4
        if self.use_pca:
            if not os.path.exists(self.pca_model_path):
                self.pca_processor.fit_incremental(
                    dataset, 
                    batch_size=batch_size, 
                    num_workers=num_workers, 
                    checkpoint_path=pca_ckpt_path, 
                    save_interval=100
                )
                self.pca_processor.save(self.pca_model_path)
                if os.path.exists(pca_ckpt_path):
                    os.remove(pca_ckpt_path)
            else:
                self.pca_processor.load(self.pca_model_path)
        existing_chunks = [f for f in os.listdir(feat_chunks_dir) if f.startswith("chunk_") and f.endswith(".pt")]
        start_batch_idx = 0

        if existing_chunks:
            indices = sorted([int(f.split('_')[1].split('.')[0]) for f in existing_chunks])
            expected_idx = 0
            valid_until = -1
            for idx in indices:
                if idx == expected_idx:
                    valid_until = idx
                    expected_idx += 1
                else:
                    break # 連続性が途切れたらそこで終了
            
            if valid_until >= 0:
                start_batch_idx = valid_until + 1
                print(f"Resuming feature transformation from batch {start_batch_idx} (contiguous).")
            else:
                print("Found chunks but sequence is broken from the start. Starting from 0.")
                start_batch_idx = 0

        print("Transforming all data...")
        total_samples = len(dataset)
        start_sample_idx = start_batch_idx * batch_size
        
        if start_sample_idx < total_samples:
            remaining_indices = range(start_sample_idx, total_samples)
            subset = Subset(dataset, remaining_indices)
            transform_loader = DataLoader(subset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
            
            for i_local, batch in enumerate(tqdm(transform_loader, desc="Transforming")):
                current_batch_idx = start_batch_idx + i_local
                
                imgs = batch["image"]
                flat = imgs.view(imgs.shape[0], -1) 

                if self.use_pca:
                    feat = self.pca_processor.transform(flat) 
                else:
                    feat = flat

                chunk_filename = f"chunk_{current_batch_idx:06d}.pt"
                chunk_path = os.path.join(feat_chunks_dir, chunk_filename)
                chunk_tmp = chunk_path + ".tmp"
                torch.save(feat.cpu(), chunk_tmp)
                os.replace(chunk_tmp, chunk_path)
        else:
            print("All chunks seem to be already processed.")
        print("Merging feature chunks...")
        all_chunk_files = sorted([os.path.join(feat_chunks_dir, f) for f in os.listdir(feat_chunks_dir) if f.startswith("chunk_")])
        
        if not all_chunk_files:
            raise RuntimeError("No feature chunks found. Something went wrong.")
        first_chunk = torch.load(all_chunk_files[0])
        chunk_size = first_chunk.shape[0]
        feature_dim = first_chunk.shape[1]
        total_samples_calc = 0
        for fpath in all_chunk_files:
             pass
        last_chunk = torch.load(all_chunk_files[-1])
        total_samples_calc = (len(all_chunk_files) - 1) * batch_size + last_chunk.shape[0]
        
        if total_samples_calc != total_samples:
            print(f"Warning: Calculated samples ({total_samples_calc}) != Dataset samples ({total_samples}).")
            print("This might happen if dataset size changed or batch_size changed between runs.")

        print(f"Total samples: {total_samples_calc}, Dimension: {feature_dim}")
        full_tensor = torch.empty((total_samples_calc, feature_dim), dtype=first_chunk.dtype)
        
        current_idx = 0
        for fpath in tqdm(all_chunk_files, desc="Merging into Tensor"):
            chunk = torch.load(fpath)
            end_idx = current_idx + chunk.shape[0]
            full_tensor[current_idx:end_idx] = chunk
            current_idx = end_idx
        
        print(f"Saving final features cache to {self.features_cache_path}")
        torch.save(full_tensor, self.features_cache_path)
        
        print("Cleaning up temp chunks...")
        shutil.rmtree(feat_chunks_dir)
        
        full_tensor.share_memory_()
        return full_tensor
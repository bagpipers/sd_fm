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
    大規模なデータセットに対して、バッチごとにPCAを適用・更新します。
    """
    def __init__(self, n_components: int, device: str):
        """
        OnlinePCAProcessorを初期化します。

        Args:
            n_components (int): PCAで保持する主成分の数（圧縮後の次元数）。
            device (str): 変換後のテンソルを配置するデバイス（'cuda' または 'cpu'）。
        """
        self.n_components = n_components
        self.device = device
        self.pca = IncrementalPCA(n_components=n_components)
        self.is_fitted = False

    def fit_incremental(self, dataset: Dataset, batch_size: int, num_workers: int, checkpoint_path: Optional[str] = None, save_interval: int = 5000):
        """
        データセットを用いてIncremental PCAを学習させます。
        チェックポイント機能により、中断された学習を再開することが可能です。

        Args:
            dataset (Dataset): 学習に使用する画像データセット。
            batch_size (int): PCAのpartial_fitに渡すバッチサイズ。
            num_workers (int): DataLoaderのワーカー数。
            checkpoint_path (Optional[str], optional): 学習途中の状態を保存/ロードするパス。
            save_interval (int, optional): 何サンプル処理するごとにチェックポイントを保存するか。

        Returns:
            None
        """
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
        
        remaining_indices = range(int(samples_seen), total_samples)
        if len(remaining_indices) == 0:
             self.is_fitted = True
             return

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
            
            # IncrementalPCAは、特徴量次元数以上のサンプル数がバッチに必要
            if buffer_size >= min_batch_size:
                X_batch = np.concatenate(buffer_list, axis=0)
                self.pca.partial_fit(X_batch)
                buffer_list = []
                buffer_size = 0
                
                if checkpoint_path and samples_since_save >= save_interval:
                    self._save_checkpoint(checkpoint_path)
                    samples_since_save = 0
        
        # 残りのバッファを処理
        if buffer_size > 0:
            X_batch = np.concatenate(buffer_list, axis=0)
            # 既に学習済みサンプルがある場合、またはバッファが成分数以上の場合のみfit
            if buffer_size >= self.n_components or self.pca.n_samples_seen_ > 0:
                 self.pca.partial_fit(X_batch)
        
        # 学習完了後のクリーンアップ
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print(f"Cleanup: Removed intermediate PCA checkpoint {checkpoint_path}")
            except OSError as e:
                print(f"Warning: Failed to remove checkpoint: {e}")

        self.is_fitted = True
        print("PCA fitting complete.")

    def _save_checkpoint(self, path):
        """
        学習中のPCAモデルをアトミックに保存します。書き込み中の破損を防ぎます。

        Args:
            path (str): 保存先のファイルパス。
        """
        temp_path = path + ".tmp"
        joblib.dump(self.pca, temp_path)
        if os.path.exists(temp_path):
            os.replace(temp_path, path)

    def transform(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        入力テンソルをPCA変換し、GPUテンソルとして返します。
        主に学習時や推論時に使用します。

        Args:
            x_tensor (torch.Tensor): 入力画像テンソル [Batch, C, H, W] または [Batch, FlatDim]。

        Returns:
            torch.Tensor: PCA変換後の特徴量テンソル [Batch, n_components]。デバイスはself.device。
        
        Raises:
            RuntimeError: PCAがまだ学習（fit）されていない場合。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        
        x_cpu = x_tensor.detach().cpu().view(x_tensor.shape[0], -1).numpy()
        x_pca = np.dot(x_cpu, self.pca.components_.T)
        
        return torch.from_numpy(x_pca).to(self.device).float()

    def transform_numpy(self, x_numpy: np.ndarray) -> np.ndarray:
        """
        NumPy配列を入力として受け取り、PCA変換後のNumPy配列を返します。
        データ前処理（特徴量抽出）用であり、GPU転送を行わないためディスク保存処理などで高速です。

        Args:
            x_numpy (np.ndarray): 入力画像データ [Batch, FlatDim]。

        Returns:
            np.ndarray: PCA変換後の特徴量データ [Batch, n_components]。
        
        Raises:
            RuntimeError: PCAがまだ学習（fit）されていない場合。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return np.dot(x_numpy, self.pca.components_.T)

    def get_components_tensor(self) -> torch.Tensor:
        """
        PCAの基底ベクトル（components_）をTensorとして取得します。

        Returns:
            torch.Tensor: PCA成分行列 [n_components, OriginalDim]。デバイスはself.device。
        
        Raises:
            RuntimeError: PCAがまだ学習（fit）されていない場合。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA is not fitted yet.")
        return torch.from_numpy(self.pca.components_).float().to(self.device)

    def save(self, path: str):
        """
        PCAモデルをjoblib形式で保存します。

        Args:
            path (str): 保存先のパス。
        """
        joblib.dump(self.pca, path)
        print(f"PCA model saved to {path}")

    def load(self, path: str):
        """
        PCAモデルをjoblib形式からロードします。

        Args:
            path (str): ロード元のパス。
        """
        self.pca = joblib.load(path)
        self.is_fitted = True
        print(f"PCA model loaded from {path}")


class SD_Manager:
    """
    SD-FM (Semidiscrete Flow Matching) の学習に必要な前処理を一括管理するクラス。
    特徴量抽出、PCA学習、ポテンシャル（双対変数）の学習、およびデータローダーの構築を担当します。
    """
    def __init__(self, config: dict, device: str):
        """
        SD_Managerを初期化します。保存ディレクトリやパスの設定を行います。

        Args:
            config (dict): 設定ファイルの内容を含む辞書。
            device (str): 計算に使用するデバイス。
        """
        self.config = config
        self.sd_config = config.get('sd_config', {})
        self.device = device
        
        self.save_dir = config['training'].get('save_dir', 'outputs')
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 各種ファイルのパス設定
        self.potential_path = os.path.join(self.save_dir, "sd_potential.pt")
        self.potential_ckpt_path = os.path.join(self.save_dir, "sd_potential_checkpoint.pt")
        self.pca_model_path = os.path.join(self.save_dir, "pca_model.joblib")
        self.features_mmap_path = os.path.join(self.save_dir, "features.mmap")
        self.features_done_flag = os.path.join(self.save_dir, "features_done.flag")
        
        self.raw_dim = config['data']['height'] * config['data']['width'] * config['data']['channels']
        self.use_pca = self.sd_config.get('use_pca', False)
        self.feature_dim = self.sd_config['pca_dim'] if self.use_pca else self.raw_dim
        
        if self.use_pca:
            self.pca_processor = OnlinePCAProcessor(self.feature_dim, device)
        else:
            self.pca_processor = None

    def prepare_dataloader(self, raw_dataset: Dataset) -> DataLoader:
        """
        SD-FM学習に必要なデータローダーを準備します。
        以下の手順を順次実行します：
        1. 特徴量の抽出（必要であればPCA学習含む）とディスクへのキャッシュ。
        2. Semidiscrete OT のポテンシャル $g$ の学習（Phase 1）。
        3. ペアリング機能を持つ専用データセット（SemidiscretePairingDataset）の構築。

        Args:
            raw_dataset (Dataset): 元の画像データセット。

        Returns:
            DataLoader: (ノイズ, 画像) の最適ペアを生成するデータローダー。
        """
        print("\n=== [SD-FM Manager] Preparing Data & Potential... ===")
        features_tensor = self._prepare_features(raw_dataset)
        potential_g = self._prepare_potential(features_tensor, len(raw_dataset))
        
        pca_components_tensor = None
        if self.use_pca and self.pca_processor is not None:
            pca_components_tensor = self.pca_processor.get_components_tensor()
            
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            dataset_features=features_tensor, # CPU上のmmap Tensor
            potential_g=potential_g.cpu(),    # ポテンシャルもCPUへ
            device=torch.device('cpu'),       # 明示的にCPU指定（ペアリング計算用）
            batch_size=self.config['training']['batch_size'],
            pca_components=pca_components_tensor.cpu() if pca_components_tensor is not None else None,
            chunk_size=self.sd_config.get('pairing_chunk_size', 10000)
        )
        
        num_workers = self.config['training'].get('num_workers', 0)
        print(f"SD-FM Dataloader created. (num_workers={num_workers})")
        
        return DataLoader(
            sd_dataset, 
            batch_size=None,   # Dataset側でバッチ化済みのためNone
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=True
        )
    
    def _prepare_potential(self, features_tensor: torch.Tensor, dataset_size: int) -> torch.Tensor:
        """
        Semidiscrete OT のポテンシャル $g$ を学習、または学習済みファイルからロードします。
        
        Args:
            features_tensor (torch.Tensor): データセットの特徴量テンソル [N, Feature_Dim]。
            dataset_size (int): データセットの総サンプル数 N。

        Returns:
            torch.Tensor: 学習済みのポテンシャルベクトル $g$ [N]。
        """
        if os.path.exists(self.potential_path):
            print(f"Loading finished potential from {self.potential_path}")
            try:
                data = torch.load(self.potential_path, map_location=self.device)
                if isinstance(data, dict) and 'g_ema' in data:
                    return data['g_ema']
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
            save_interval=5000 # 5000ステップごとに保存
        )
        torch.save(g_ema, self.potential_path)
        print(f"Potential training finished and saved to {self.potential_path}")
        if os.path.exists(self.potential_ckpt_path):
            os.remove(self.potential_ckpt_path)
            
        return g_ema

    def _prepare_features(self, dataset: Dataset) -> torch.Tensor:
        """
        データセット内の全画像から特徴量を抽出し、NumPy memmap形式でディスクにキャッシュします。
        PCAが有効な場合は、特徴量抽出前にPCAの学習（fit）も行います。
        
        メモリストレージ（memmap）を使用することで、大規模データセットでもRAMを圧迫せずに処理可能です。

        Args:
            dataset (Dataset): 元の画像データセット。

        Returns:
            torch.Tensor: キャッシュされた特徴量をラップしたTensor [N, Feature_Dim]。
                          実体はディスク上のmemmapファイルです。
        """
        total_samples = len(dataset)
        
        # 既に完了フラグとファイルが存在する場合はロードして終了
        if os.path.exists(self.features_mmap_path) and os.path.exists(self.features_done_flag):
            if self.use_pca and not self.pca_processor.is_fitted:
                if os.path.exists(self.pca_model_path):
                    print("Loading PCA model for cached features...")
                    self.pca_processor.load(self.pca_model_path)
            
            print(f"Loading cached features via mmap from {self.features_mmap_path}")
            mmap_features = np.memmap(self.features_mmap_path, dtype='float32', mode='r', shape=(total_samples, self.feature_dim))
            return torch.from_numpy(mmap_features)

        print("Extracting features from dataset...")
        
        # PCAの学習 (必要な場合)
        if self.use_pca:
            if not os.path.exists(self.pca_model_path):
                fit_batch_size = 256 
                self.pca_processor.fit_incremental(
                    dataset, 
                    batch_size=fit_batch_size, 
                    num_workers=0, 
                    checkpoint_path=os.path.join(self.save_dir, "pca_checkpoint_fit.joblib")
                )
                self.pca_processor.save(self.pca_model_path)
            else:
                self.pca_processor.load(self.pca_model_path)
        
        # 特徴量抽出の進捗管理
        progress_path = os.path.join(self.save_dir, "features_progress.json")
        processed_count = 0
        if os.path.exists(progress_path):
            try:
                with open(progress_path, 'r') as f:
                    data = json.load(f)
                    processed_count = data.get("processed_count", 0)
                print(f"Resuming feature extraction from sample {processed_count}...")
            except:
                print("Failed to load progress file. Starting from 0.")
        
        # memmapのモード設定
        if processed_count > 0 and os.path.exists(self.features_mmap_path):
             mode = 'r+'
        else:
             mode = 'w+'
             processed_count = 0 

        mmap_features = np.memmap(
            self.features_mmap_path, 
            dtype='float32', 
            mode=mode, 
            shape=(total_samples, self.feature_dim)
        )
        
        # 未処理分のデータを処理
        if processed_count < total_samples:
            subset = Subset(dataset, range(processed_count, total_samples))
            transform_batch_size = 256
            loader = DataLoader(subset, batch_size=transform_batch_size, num_workers=4, shuffle=False)
            
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
                    with open(progress_path, 'w') as f:
                        json.dump({"processed_count": current_idx}, f)

            mmap_features.flush()
        
        print("Feature extraction complete.")
        with open(self.features_done_flag, 'w') as f:
            f.write("done")
            
        if os.path.exists(progress_path):
            os.remove(progress_path)
        
        # 書き込みモードから読み込みモードへ再オープン
        del mmap_features
        mmap_features = np.memmap(self.features_mmap_path, dtype='float32', mode='r', shape=(total_samples, self.feature_dim))
        
        return torch.from_numpy(mmap_features)
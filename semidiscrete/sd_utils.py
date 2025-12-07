import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from .sd_solver import SemidiscreteOT_Solver
from .sd_loader import SemidiscretePairingDataset

class SD_Manager:
    """
    TrainスクリプトとSDロジックの仲介役。
    - ポテンシャルの有無を確認・学習・ロード
    - データセットのフラット化（VRAM展開）
    - SD専用データローダーの構築
    を一手に引き受けます。
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.save_dir = config['training'].get('save_dir', 'outputs')
        self.potential_path = os.path.join(self.save_dir, "sd_potential.pt")
        
        # 画像サイズから特徴量次元を計算 (C*H*W)
        self.feature_dim = config['data']['height'] * config['data']['width'] * config['data']['channels']

    def prepare_dataloader(self, raw_dataset):
        """
        raw_dataset (T2IDataset) を受け取り、
        ペアリング済みの IterableDataLoader を返す。
        必要であればポテンシャルの学習(Phase 1)を実行する。
        """
        print("\n=== [SD-FM Manager] Preparing Data... ===")
        
        # 1. データセットをフラット化してメモリに展開 (Solver/Loader両方で必要)
        # ※本来はキャッシュ推奨だが、ここでは簡易化のため毎回変換またはメモリ保持
        flattened_data = self._flatten_dataset(raw_dataset)
        
        # 2. ポテンシャルの準備
        if os.path.exists(self.potential_path):
            print(f"Loading existing potential from {self.potential_path}")
            g_ema = torch.load(self.potential_path, map_location=self.device)
        else:
            print("Potential not found. Starting training...")
            solver = SemidiscreteOT_Solver(
                dataset_size=len(raw_dataset),
                feature_dim=self.feature_dim,
                device=self.device
            )
            g_ema = solver.train_loop(flattened_data)
            torch.save(g_ema, self.potential_path)
            print(f"Potential saved to {self.potential_path}")

        # 3. 専用ローダーの構築
        sd_dataset = SemidiscretePairingDataset(
            original_dataset=raw_dataset,
            potential_g=g_ema,
            all_images_flat=flattened_data,
            feature_dim=self.feature_dim,
            device=self.device,
            batch_size=self.config['training']['batch_size']
        )
        
        # IterableDatasetなので num_workers=0 (GPU tensorを直接扱うためメインプロセス推奨)
        # batch_size=None は Dataset側でバッチを作るため必須
        return DataLoader(sd_dataset, batch_size=None, num_workers=0)

    def _flatten_dataset(self, dataset):
        print("Flattening dataset for SD calculations...")
        # メモリ効率のため、データローダー経由で収集
        temp_loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=False)
        all_tensors = []
        for batch in tqdm(temp_loader, desc="Flattening"):
            imgs = batch["image"] # T2IDatasetの戻り値キーに合わせる (getitem参照)
            all_tensors.append(imgs.view(imgs.shape[0], -1))
        
        full_tensor = torch.cat(all_tensors, dim=0)
        print(f"Dataset flattened: {full_tensor.shape}")
        return full_tensor
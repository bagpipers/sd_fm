import torch
from torch.utils.data import IterableDataset
import math

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM用のデータセットラッパー。
    学習済みポテンシャル g を使い、ランダムなノイズ x に対して
    最適なデータ y (および条件 c) を割り当てて返す。
    
    これにより「画像とテキストの対応関係」を壊さずに、
    「ノイズと画像の最適な幾何学的ペアリング」を実現する。
    """
    def __init__(self, 
                 original_dataset, 
                 potential_path: str,
                 feature_dim: int,
                 device: torch.device,
                 batch_size: int = 64):
        
        self.dataset = original_dataset
        self.device = device
        self.batch_size = batch_size
        self.feature_dim = feature_dim
        
        # 学習済みポテンシャル g のロード
        print(f"Loading potential from {potential_path}...")
        self.g = torch.load(potential_path, map_location=device)
        self.g.requires_grad_(False)
        
        # データセット全体をメモリに展開 (Precompute同様、高速検索のため)
        # 注意: ImageNet等の超大規模データの場合は、Faiss等のMIPSライブラリを使うか
        # Datasetをオンメモリに置く工夫が必要。
        print("Loading full dataset into memory for pairing...")
        self.all_images_flat = self._load_all_images(original_dataset).to(device)
        
        # 事前計算: ||y||^2
        self.Y_sq_norm = torch.sum(self.all_images_flat ** 2, dim=1)

    def _load_all_images(self, dataset):
        # データセットから画像だけを取り出して [M, D] にする
        # ※ dataset[i] が重い処理を含む場合、ここは時間がかかります
        images = []
        for i in range(len(dataset)):
            item = dataset[i]
            img = item["image"] # PIL or Tensor
            if not isinstance(img, torch.Tensor):
                 # Transformがdataset内で適用されていない場合の保険
                 # 実際は T2IDataset で transform 済み Tensor が返る想定
                 pass 
            # [C, H, W] -> [D]
            images.append(img.view(-1))
        return torch.stack(images)

    def __iter__(self):
        return self.generator()

    def generator(self):
        """
        無限にバッチを生成するジェネレータ
        """
        while True:
            # 1. ノイズバッチ X ~ N(0, I)
            X = torch.randn(self.batch_size, self.feature_dim, device=self.device)
            
            # 2. 最適なペア探索 (ASSIGN Algorithm)
            # Score = g_j - ||y_j||^2 + 2 <x, y_j>
            # 行列積: [B, D] @ [D, M] -> [B, M]
            # ※ データ数Mが多い場合、ここをチャンク分割して計算する必要があります
            cross_term = 2 * torch.matmul(X, self.all_images_flat.t())
            bias = self.g - self.Y_sq_norm
            scores = cross_term + bias.unsqueeze(0)
            
            # argmax でインデックス取得
            # indices: [B] (各ノイズに対応するデータセット内のインデックス)
            indices = torch.argmax(scores, dim=1).cpu().numpy()
            
            # 3. ペアデータの構築 (ここが重要)
            # 取得したインデックスを使って、オリジナルのDatasetから
            # 「画像」と「テキスト(条件)」をセットで取得する。
            batch_images = []
            batch_pos_prompts = []
            batch_neg_prompts = []
            
            # 割り当てられたノイズはそのまま使う
            batch_noise = X
            
            for idx in indices:
                # オリジナルデータセットから取得 (画像とテキストの対応は維持される)
                item = self.dataset[int(idx)]
                
                # 画像 (ターゲット y1)
                batch_images.append(item["image"])
                
                # テキスト条件
                batch_pos_prompts.append(item["positive_prompt"])
                batch_neg_prompts.append(item["negative_prompt"])
            
            # スタックして返す
            pixel_values = torch.stack(batch_images)
            
            yield {
                "pixel_values": pixel_values,      # y1 (Data)
                "noise": batch_noise,              # x0 (Matched Noise)
                "positive_prompt": batch_pos_prompts,
                "negative_prompt": batch_neg_prompts
            }

    def __len__(self):
        # IterableDataset なので近似的な長さを返す
        return len(self.dataset)
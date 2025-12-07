import torch
from torch.utils.data import IterableDataset

class SemidiscretePairingDataset(IterableDataset):
    """
    SD-FM用データセット。
    1. ランダムノイズ x を生成
    2. 学習済みポテンシャル g を用いて最適な画像インデックス k を決定
    3. オリジナルデータセットから k 番目の (画像, プロンプト) を取得して返す
    """
    def __init__(self, original_dataset, potential_g, all_images_flat, feature_dim, device, batch_size=64):
        self.dataset = original_dataset
        self.g = potential_g.to(device)
        self.all_images_flat = all_images_flat.to(device)
        self.Y_sq_norm = torch.sum(self.all_images_flat ** 2, dim=1)
        
        self.feature_dim = feature_dim
        self.device = device
        self.batch_size = batch_size

    def __iter__(self):
        while True:
            # 1. ノイズ生成 (Source)
            X = torch.randn(self.batch_size, self.feature_dim, device=self.device)
            
            # 2. マッチング (Target Index Search)
            # MIPS (Maximum Inner Product Search) + Bias
            cross_term = 2 * torch.matmul(X, self.all_images_flat.t())
            bias = self.g - self.Y_sq_norm
            scores = cross_term + bias.unsqueeze(0)
            
            indices = torch.argmax(scores, dim=1).cpu().numpy()
            
            # 3. データ取得 (Correspondence 維持)
            batch_pixels = []
            batch_pos = []
            batch_neg = []
            
            for idx in indices:
                item = self.dataset[int(idx)] # ここで画像とテキストのペアを取得
                batch_pixels.append(item["image"]) # tensor
                batch_pos.append(item["positive_prompt"])
                batch_neg.append(item["negative_prompt"])
                
            yield {
                "pixel_values": torch.stack(batch_pixels), # x_1 (Target Image)
                "noise": X,                                # x_0 (Paired Noise)
                "positive_prompt": batch_pos,
                "negative_prompt": batch_neg
            }
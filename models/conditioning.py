import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Dict, List

class ConditioningModel(nn.Module):
    """
    config.yaml の 'condition_type' に基づき、
    'clip' (テキスト) または 'class' (nn.Embedding) の
    条件付け埋め込みを生成するモジュール。
    
    CFG (Classifier-Free Guidance) のための
    「条件ドロップ」ロジックも内包します。
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.model_cfg = config['model']
        self.condition_type = self.model_cfg['condition_type']
        self.cfg_drop_prob = config['training']['cfg_drop_prob']
        
        print(f"ConditioningModel: Initializing in '{self.condition_type}' mode.")

        if self.condition_type == "clip":
            # --- 1. CLIP モード ---
            clip_cfg = self.model_cfg['clip_config']
            self.embed_dim = clip_cfg['embed_dim']
            self.max_length = clip_cfg['max_text_length']
            
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_cfg['text_encoder_name'])
            self.text_encoder = CLIPTextModel.from_pretrained(clip_cfg['text_encoder_name'])
            
            self.uncond_embedding = nn.Parameter(
                torch.randn(1, self.max_length, self.embed_dim)
            )
            
            self.text_encoder.eval()
            self.text_encoder.requires_grad_(False)

        elif self.condition_type == "class":
            # --- 2. Class モード (MNIST用) ---
            class_cfg = self.model_cfg['class_config']
            self.num_classes = class_cfg['num_classes']
            self.embed_dim = class_cfg['embed_dim']
            
            self.embedding = nn.Embedding(self.num_classes + 1, self.embed_dim)
            self.uncond_class_index = self.num_classes
        
        else:
            raise ValueError(f"Unknown condition_type: {self.condition_type}")

    def forward(self, batch: Dict, device: torch.device) -> torch.Tensor:
        """
        Dataloaderから受け取ったバッチを処理し、
        CFGドロップを適用した後の条件付け埋め込み
        [B, Seq, Dim] を返す。
        
        self.training == True (訓練時):
            cfg_drop_prob に基づき、ランダムにドロップ (無条件化) する。
        self.training == False (推論時):
            ドロップせず、条件付き埋め込みのみを返す。
        """
        b = batch["pixel_values"].shape[0]
        
        # --- ★ 修正点: self.training (eval()モード) かどうかをチェック ---
        if self.training:
            # (訓練時) CFG: ドロップするかどうかのマスクをランダム生成
            drop_mask = (torch.rand(b, device=device) < self.cfg_drop_prob)
        else:
            # (推論時) ドロップしない (すべて条件付き)
            drop_mask = torch.zeros(b, device=device, dtype=torch.bool)
        # --- ★ 修正ここまで ---

        if self.condition_type == "clip":
            return self._forward_clip(batch, drop_mask, device)
        elif self.condition_type == "class":
            return self._forward_class(batch, drop_mask, device)

    # --- ★ 追加メソッド: 無条件埋め込みを明示的に取得 ---
    @torch.no_grad()
    def get_uncond_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        CFG推論 (sample.py) のために、
        無条件 (Unconditional) 埋め込み [B, Seq, Dim] を返す。
        """
        if self.condition_type == "clip":
            # [1, 77, 768] -> [B, 77, 768]
            return self.uncond_embedding.expand(batch_size, -1, -1)
        
        elif self.condition_type == "class":
            # 無条件クラスインデックス [B]
            labels = torch.full(
                (batch_size,), 
                self.uncond_class_index, 
                dtype=torch.long, 
                device=device
            )
            # [B] -> [B, embed_dim] -> [B, 1, embed_dim]
            return self.embedding(labels).unsqueeze(1)
    # --- ★ 追加ここまで ---

    def _forward_clip(self, batch: Dict, drop_mask: torch.Tensor, device: torch.device):
        """ 'clip' モードのフォワード + CFG """
        
        # (1) テキストプロンプトをトークン化 (dataloader.py からの生テキスト)
        prompts = batch["positive_prompt"]
        
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        
        # (2) テキストエンコーダで埋め込みを計算
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask
            ).last_hidden_state # [B, 77, 768]
            
        # (3) CFG: ドロップ対象を「無条件埋め込み」に置き換え
        uncond_embeds = self.uncond_embedding.expand(b, -1, -1)
        
        mask_view = drop_mask.view(-1, 1, 1)
        final_embeddings = torch.where(mask_view, uncond_embeds, text_embeddings)
        
        return final_embeddings

    def _forward_class(self, batch: Dict, drop_mask: torch.Tensor, device: torch.device):
        """ 'class' モードのフォワード + CFG """
        
        # (1) プロンプト文字列 (例: ["1", "5", "1"]) を
        #     整数のラベル (例: [1, 5, 1]) に変換
        try:
            labels = torch.tensor(
                [int(p) for p in batch["positive_prompt"]], 
                dtype=torch.long, 
                device=device
            )
        except ValueError:
            raise ValueError(f"In 'class' mode, positive_prompt must be convertible to int. Got: {batch['positive_prompt']}")
            
        # (2) CFG: ドロップ対象を「無条件クラスインデックス」に置き換え
        uncond_label = self.uncond_class_index
        
        final_labels = torch.where(drop_mask, uncond_label, labels)
        
        # (3) nn.Embedding を通して埋め込みベクトルを取得
        final_embeddings = self.embedding(final_labels)
        
        # (4) Cross-Attention (UNet) が要求する [B, Seq, Dim] の
        #     形式に合わせるため、次元を追加
        return final_embeddings.unsqueeze(1)
import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from typing import Dict, List

class ConditioningModel(nn.Module):
    """
    config.yaml の 'condition_type' に基づき、条件付け埋め込みを生成するモジュール。
    
    [修正点]
    'class' モード時、configの num_classes に対して自動的に2つのクラスを追加する。
      - index = num_classes     : Trash Class (破損画像用、例: 1000)
      - index = num_classes + 1 : Unconditional Token (CFG用、例: 1001)
    これにより、ユーザーはconfigを書き換える必要がない。
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.model_cfg = config['model']
        self.condition_type = self.model_cfg['condition_type']
        self.cfg_drop_prob = config['training']['cfg_drop_prob']
        
        print(f"ConditioningModel: Initializing in '{self.condition_type}' mode.")

        if self.condition_type == "clip":
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
            class_cfg = self.model_cfg['class_config']
            self.num_classes = class_cfg['num_classes'] 
            self.embed_dim = class_cfg['embed_dim']
            self.trash_class_index = self.num_classes
            self.uncond_class_index = self.num_classes + 1
            self.embedding = nn.Embedding(self.num_classes + 2, self.embed_dim)
            
            print(f"  - Class Embedding Size: {self.num_classes + 2}")
            print(f"  - Trash Class Index: {self.trash_class_index}")
            print(f"  - Uncond Class Index: {self.uncond_class_index}")
        
        else:
            raise ValueError(f"Unknown condition_type: {self.condition_type}")

    def forward(self, batch: Dict, device: torch.device) -> torch.Tensor:
        # [修正] train_sd.py (sd_loader) は "image" をキーとして使用するため修正
        # 元コード: b = batch["pixel_values"].shape[0]
        b = batch["image"].shape[0]

        if self.training:
            drop_mask = (torch.rand(b, device=device) < self.cfg_drop_prob)
        else:
            drop_mask = torch.zeros(b, device=device, dtype=torch.bool)

        if self.condition_type == "clip":
            return self._forward_clip(batch, drop_mask, device)
        elif self.condition_type == "class":
            return self._forward_class(batch, drop_mask, device)

    @torch.no_grad()
    def get_uncond_embedding(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if self.condition_type == "clip":
            return self.uncond_embedding.expand(batch_size, -1, -1)
        
        elif self.condition_type == "class":
            labels = torch.full(
                (batch_size,), 
                self.uncond_class_index, # 例: 1001
                dtype=torch.long, 
                device=device
            )
            return self.embedding(labels).unsqueeze(1)

    def _forward_clip(self, batch: Dict, drop_mask: torch.Tensor, device: torch.device):
        prompts = batch["positive_prompt"]
        b = len(prompts)
        
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask
            ).last_hidden_state 
        uncond_embeds = self.uncond_embedding.expand(b, -1, -1)
        
        mask_view = drop_mask.view(-1, 1, 1)
        final_embeddings = torch.where(mask_view, uncond_embeds, text_embeddings)
        
        return final_embeddings

    def _forward_class(self, batch: Dict, drop_mask: torch.Tensor, device: torch.device):
        if "class_id" in batch:
            labels = batch["class_id"].to(device, dtype=torch.long)
        else:
            try:
                labels = torch.tensor(
                    [int(p) for p in batch["positive_prompt"]], 
                    dtype=torch.long, 
                    device=device
                )
            except ValueError:
                raise KeyError(
                    "condition_type='class' is selected, but 'class_id' is missing in batch "
                    "and 'positive_prompt' contains non-integer strings."
                )
        
        uncond_label = self.uncond_class_index
        
        final_labels = torch.where(drop_mask, uncond_label, labels)
        final_embeddings = self.embedding(final_labels)
        return final_embeddings.unsqueeze(1)
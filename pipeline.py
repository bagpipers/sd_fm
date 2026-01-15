import torch
import yaml
import os
from typing import List, Optional, Dict
from models.unet import TextConditionedUNet
from models.conditioning import ConditioningModel
from solvers.ode_solvers import SimpleODESolver

class InferencePipeline:
    """
    Text-to-Image 推論を実行するためのパイプライン。
    モデルのロードと推論ロジックをカプセル化。
    """
    def __init__(self, 
                 config_path: str = "configs/config_sd.yaml", 
                 checkpoint_path: str = "outputs/model.pth", 
                 device: Optional[str] = None):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイル {config_path} が見つかりません。")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() and self.config['device'] == 'cuda' else "cpu"
        
        print(f"Pipeline: Using device: {self.device}")
        self.model = TextConditionedUNet(self.config).to(self.device)
        self.condition_model = ConditioningModel(self.config).to(self.device)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"モデルファイル {checkpoint_path} が見つかりません。")
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' not in checkpoint:
            try:
                self.model.load_state_dict(checkpoint)
            except:
                 raise KeyError(f"チェックポイントに 'model_state_dict' が見つかりません。")
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'condition_model_state_dict' in checkpoint:
            self.condition_model.load_state_dict(checkpoint['condition_model_state_dict'])
        else:
            print("Warning: 'condition_model_state_dict' not found in checkpoint. Using initialized weights.")

        self.model.eval()
        self.condition_model.eval()
        self.solver = SimpleODESolver(self.model)
        
        print(f"Pipeline: Models loaded and set to eval mode.")

    @torch.no_grad()
    def __call__(
        self, 
        prompts: List[str], 
        guidance_scale: Optional[float] = None, 
        steps: Optional[int] = None, 
        method: Optional[str] = None
    ) -> torch.Tensor:
        """
        プロンプトのリストを受け取り、生成された画像のテンソルを返す。
        """
        if not prompts:
            print("Warning: prompts リストが空です。何も生成しません。")
            return torch.empty(0)

        sampling_cfg = self.config['sampling']
        cfg_scale = guidance_scale if guidance_scale is not None else sampling_cfg.get('guidance_scale', 0.0)
        num_steps = steps if steps is not None else sampling_cfg.get('steps', 20)
        solve_method = method if method is not None else sampling_cfg.get('method', 'euler')

        batch_size = len(prompts)
        dummy_images = torch.empty(batch_size, 0, 0, 0, device=self.device) 
        
        dummy_batch = {
            "positive_prompt": prompts,
            "image": dummy_images  # [修正] キーを "pixel_values" から "image" に変更
        }
        cond = self.condition_model(dummy_batch, self.device)
        
        uncond = None
        if cfg_scale > 0.0:
            uncond = self.condition_model.get_uncond_embedding(batch_size, self.device)
            
        data_cfg = self.config['data']
        x_init = torch.randn(
            batch_size,
            data_cfg['channels'],
            data_cfg['height'], 
            data_cfg['width']   
        ).to(self.device)
        generated_images = self.solver.sample(
            x_init, 
            cond, 
            uncond_condition=uncond,
            guidance_scale=cfg_scale,
            steps=num_steps, 
            method=solve_method
        )
        
        return generated_images
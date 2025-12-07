import torch
import yaml
import os
from typing import List, Optional, Dict

# 既存モジュールのインポート
from models.unet import TextConditionedUNet
from models.conditioning import ConditioningModel
from solvers.ode_solvers import SimpleODESolver

class InferencePipeline:
    """
    Text-to-Image 推論を実行するためのパイプライン。
    モデルのロードと推論ロジックをカプセル化します。
    
    使用例:
        pipe = InferencePipeline(config_path="configs/config.yaml", checkpoint_path="outputs/model.pth")
        images_tensor = pipe(["a photo of a cat", "a photo of a dog"])
        save_image(images_tensor, "output.png", normalize=True, value_range=(-1, 1))
    """
    def __init__(self, 
                 config_path: str = "configs/config.yaml", 
                 checkpoint_path: str = "outputs/model.pth", 
                 device: Optional[str] = None):
        
        # 1. 設定の読み込み
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"設定ファイル {config_path} が見つかりません。")
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 2. デバイス設定
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() and self.config['device'] == 'cuda' else "cpu"
        print(f"Pipeline: Using device: {self.device}")

        # 3. モデルの準備 (UNet と ConditionModel)
        self.model = TextConditionedUNet(self.config).to(self.device)
        self.condition_model = ConditioningModel(self.config).to(self.device)
        
        # 4. チェックポイントのロード
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"モデルファイル {checkpoint_path} が見つかりません。")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"チェックポイントに 'model_state_dict' (UNet) が見つかりません。")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'condition_model_state_dict' not in checkpoint:
            raise KeyError(f"チェックポイントに 'condition_model_state_dict' (ConditioningModel) が見つかりません。")
        self.condition_model.load_state_dict(checkpoint['condition_model_state_dict'])
        
        # ★ 推論モードに設定
        self.model.eval()
        self.condition_model.eval()
        
        # 5. ODEソルバーの初期化
        self.solver = SimpleODESolver(self.model)
        
        print(f"Pipeline: Models loaded from {checkpoint_path} and set to eval mode.")

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

        Args:
            prompts (List[str]): 生成したいプロンプトのリスト。
                                 ("class" モードの場合は ["0", "1"] など)
            guidance_scale (Optional[float]): CFG強度。None の場合 config.yaml の値を使用。
            steps (Optional[int]): サンプリングステップ数。None の場合 config.yaml の値を使用。
            method (Optional[str]): ソルバー。None の場合 config.yaml の値を使用。

        Returns:
            torch.Tensor: 生成された画像のテンソル (B, C, H, W)。
                          [-1, 1] の範囲に正規化されています。
        """
        
        if not prompts:
            print("Warning: prompts リストが空です。何も生成しません。")
            return torch.empty(0)

        # --- 1. 推論パラメータの設定 ---
        sampling_cfg = self.config['sampling']
        
        # config.yaml の値 (デフォルト) -> 引数で上書き
        cfg_scale = guidance_scale if guidance_scale is not None else sampling_cfg.get('guidance_scale', 0.0)
        num_steps = steps if steps is not None else sampling_cfg.get('steps', 20)
        solve_method = method if method is not None else sampling_cfg.get('method', 'euler')

        batch_size = len(prompts)
        
        # --- 2. 条件埋め込み (cond) の生成 ---
        # (self.condition_model は .eval() モードなのでドロップは発生しない)
        dummy_pixel_values = torch.empty(batch_size, 0, 0, 0, device=self.device) 
        dummy_batch = {
            "positive_prompt": prompts,
            "pixel_values": dummy_pixel_values
        }
        cond = self.condition_model(dummy_batch, self.device)
        
        # --- 3. 無条件埋め込み (uncond) の生成 ---
        uncond = None
        if cfg_scale > 0.0:
            uncond = self.condition_model.get_uncond_embedding(batch_size, self.device)
        
        # --- 4. 初期ノイズ (x_init) の生成 ---
        data_cfg = self.config['data']
        
        x_init = torch.randn(
            batch_size,
            data_cfg['channels'],
            data_cfg['height'], 
            data_cfg['width']   
        ).to(self.device)

        # --- 5. サンプリング実行 ---
        print(f"Sampling {batch_size} images with {num_steps} steps (CFG: {cfg_scale})...")
        generated_images = self.solver.sample(
            x_init, 
            cond, 
            uncond_condition=uncond,
            guidance_scale=cfg_scale,
            steps=num_steps, 
            method=solve_method
        )
        
        # [-1, 1] のテンソルを返す
        return generated_images
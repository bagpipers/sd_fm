import torch
import yaml
from torchvision.utils import save_image
import os
import math

from pipeline import InferencePipeline

def main():
    # 1. 設定ファイルの読み込み (プロンプト決定とファイル名のために使用)
    config_path = "configs/config.yaml"
    checkpoint_path = "outputs/model.pth"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)


    print("Initializing inference pipeline...")
    try:
        pipe = InferencePipeline(
            config_path=config_path,
            checkpoint_path=checkpoint_path
        )
    except (FileNotFoundError, KeyError) as e:
        print(f"パイプラインの初期化に失敗しました: {e}")
        print("先に train.py を実行してモデルを保存してください。")
        return
        
    print("Pipeline initialized.")

    # --- 2. 推論用プロンプトの準備 (config に基づく) ---
    condition_type = config['model']['condition_type']
    prompts_to_sample = []
    
    if condition_type == "class":
        
        prompts_to_sample = [str(i) for i in range(10)] # 0-9
        print(f"Generating images for class prompts: {prompts_to_sample}")
    
    elif condition_type == "clip":
        
        prompts_to_sample = [
            "a high-quality photo of the digit 0", 
            "a high-quality photo of the digit 1",
            "a high-quality photo of the digit 2",
            "a high-quality photo of the digit 3",
            "a high-quality photo of the digit 4",
            "a high-quality photo of the digit 5",
            "a high-quality photo of the digit 6",
            "a high-quality photo of the digit 7",
            "a high-quality photo of the digit 8",
            "a high-quality photo of the digit 9"
        ] 
        print(f"Generating images for text prompts: {prompts_to_sample}")
    
    if not prompts_to_sample:
        print("No prompts defined for sampling. Exiting.")
        return

    # --- 3. パイプライン実行 ---
    # パイプラインは config.yaml から sampling 設定 (steps, cfg_scale) を読み込みます
    # ここで引数を渡して上書きも可能です:
    # images = pipe(prompts_to_sample, steps=50, guidance_scale=10.0)
    
    images = pipe(prompts_to_sample)
    
    print(f"Generated shape: {images.shape}")
    
    # --- 4. 画像を保存 ---
    os.makedirs("outputs", exist_ok=True)
    
    # config から CFG スケールを取得 (ファイル名用)
    guidance_scale = config['sampling'].get('guidance_scale', 0.0)
    save_path = f"outputs/generated_pipeline_{condition_type}_cfg{guidance_scale}.png"
    
    batch_size = len(prompts_to_sample)
    
    # グリッドの列数を計算 (10枚なら5列x2行)
    nrow = 5 if batch_size == 10 else math.ceil(math.sqrt(batch_size))
        
    save_image(
        images, 
        save_path, 
        nrow=nrow, 
        normalize=True, 
        value_range=(-1, 1) 
    )
    print(f"Generated images saved to {save_path}")

if __name__ == "__main__":
    main()
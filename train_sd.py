import torch
import torch.optim as optim
import yaml
import os
from torchvision import transforms 
from tqdm import tqdm 

# モデルとローダーのインポート
from models.unet import TextConditionedUNet
from models.conditioning import ConditioningModel 
from dataloader import T2IDataset 
from paths.ot_cfm_sd import PairedOTCFM       
from semidiscrete.sd_utils import SD_Manager    

def main():
    config_path = "configs/config_sd.yaml"
    if not os.path.exists(config_path):
        print(f"Warning: {config_path} not found. Falling back to configs/config.yaml")
        config_path = "configs/config.yaml"
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = config['training'].get('save_dir', 'outputs')
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loaded config from {config_path}")
    print(f"Save directory: {save_dir}")
    
    img_channels = config['data']['channels']
    h, w = config['data']['height'], config['data']['width']
    
    # 画像の前処理: [-1, 1] に正規化
    transform = transforms.Compose([
        transforms.Resize((h, w)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
    ])
    
    # 元データの準備 (SD_Managerが内部でFeature抽出に使用)
    raw_dataset = T2IDataset(
        root_dir=config['data']['root_dir'], 
        metadata_file=config['data']['metadata_file'], 
        image_transform=transform,
        target_channels=img_channels 
    )
    print(f"Raw dataset size: {len(raw_dataset)}")
    
    # SD-Manager: ポテンシャル学習とデータローダー構築
    # ※ semidiscrete/sd_loader.py が正しい(File 4)ものであることを確認してください
    sd_manager = SD_Manager(config, device)
    
    # SD-FM用データローダー (ペアリング済みバッチを返す)
    dataloader = sd_manager.prepare_dataloader(raw_dataset)
    
    # モデルの準備
    condition_model = ConditioningModel(config).to(device)
    model = TextConditionedUNet(config).to(device)
    
    # SD-FM用ロス計算クラス
    cfm = PairedOTCFM(sigma_min=1e-5)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(condition_model.parameters()), 
        lr=config['training']['lr']
    )
    
    print("\n=== Starting SD-FM Training Loop ===")
    model.train()
    condition_model.train()
    
    batch_size = config['training']['batch_size']
    # IterableDataset のため、epochあたりのステップ数を手動計算
    steps_per_epoch = len(raw_dataset) // batch_size
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs}")
        total_loss = 0
        
        for i, batch in enumerate(pbar):
            if i >= steps_per_epoch: break 
            
            # x_1: Target Data (画像) [B, C, H, W]
            images = batch["pixel_values"].to(device)
            
            # x_0: Paired Noise [B, D] (Flattened)
            # ★重要修正★: 画像と同じ形状に reshape する必要があります
            noise = batch["noise"].to(device)
            noise = noise.view_as(images)
            
            # 条件 (テキスト埋め込み)
            text_embeddings = condition_model(batch, device)
            
            optimizer.zero_grad()
            
            # SD-FM ロス計算
            # 外部から供給された最適なペア (x_0, x_1) を使用
            loss = cfm.compute_loss_paired(model, x_0=noise, x_1=images, condition=text_embeddings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # エポック終了後の保存
        epoch_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'condition_model_state_dict': condition_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, epoch_save_path)
        
        avg_loss = total_loss / steps_per_epoch if steps_per_epoch > 0 else 0
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}. Saved to {epoch_save_path}")

if __name__ == "__main__":
    main()
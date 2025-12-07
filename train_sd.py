import torch
import torch.optim as optim
import yaml
import os
from torchvision import transforms 
from tqdm import tqdm 

# 既存モジュールのインポート
from models.unet import TextConditionedUNet
from models.conditioning import ConditioningModel 
from dataloader import T2IDataset 
from paths.ot_cfm_sd import PairedOTCFM       
from semidiscrete.sd_utils import SD_Manager    

def main():
    # 1. Config & Setup
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = config['training'].get('save_dir', 'outputs')
    os.makedirs(save_dir, exist_ok=True)

    # 2. Dataset Preparation (Raw)
    img_channels = config['data']['channels']
    h, w = config['data']['height'], config['data']['width']
    transform = transforms.Compose([
        transforms.Resize((h, w)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
    ])
    raw_dataset = T2IDataset(
        root_dir=config['data']['root_dir'], 
        metadata_file=config['data']['metadata_file'], 
        image_transform=transform,
        target_channels=img_channels 
    )

    # 3. ★ SD-FM Setup (ここだけが train.py と違う) ★
    # マネージャーに丸投げすることで、PrecomputeやLoader構築の複雑さを隠蔽
    sd_manager = SD_Manager(config, device)
    dataloader = sd_manager.prepare_dataloader(raw_dataset)

    # 4. Model & Optimizer
    condition_model = ConditioningModel(config).to(device)
    model = TextConditionedUNet(config).to(device)
    
    # ★ 拡張版のCFMを使用
    cfm = PairedOTCFM(sigma_min=1e-5)
    
    optimizer = optim.Adam(
        list(model.parameters()) + list(condition_model.parameters()), 
        lr=config['training']['lr']
    )

    # 5. Training Loop
    print("Starting SD-FM Training Loop...")
    model.train()
    condition_model.train()
    
    steps_per_epoch = len(raw_dataset) // config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    global_step = 0

    for epoch in range(num_epochs):
        pbar = tqdm(dataloader, total=steps_per_epoch, desc=f"Epoch {epoch+1}")
        total_loss = 0
        
        # IterableDatasetなので enumerate ではなく直接回す
        for i, batch in enumerate(pbar):
            if i >= steps_per_epoch: break # 安全のため
            
            # SD-Loader から供給されるデータのキーを使用
            images = batch["pixel_values"].to(device) # x_1
            noise = batch["noise"].to(device)         # x_0 (Paired)
            
            # Conditioning (T2IDataset準拠のキーが入っているのでそのまま渡せる)
            text_embeddings = condition_model(batch, device)
            
            optimizer.zero_grad()
            
            # ★ ペアリング済みLoss計算
            loss = cfm.compute_loss_paired(model, x_0=noise, x_1=images, condition=text_embeddings)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            global_step += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Checkpoint Save
        torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch_{epoch+1}.pth"))
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/steps_per_epoch:.4f}")

if __name__ == "__main__":
    main()
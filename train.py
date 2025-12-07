import torch
import torch.optim as optim
import yaml
import os
from torch.utils.data import DataLoader 
from torchvision import transforms 
from tqdm import tqdm 

from models.unet import TextConditionedUNet
from models.conditioning import ConditioningModel 
from paths.ot_cfm import OTConditionalFlowMatching
from dataloader import T2IDataset, T2ICollate 

def train_step(cfm, model, optimizer, images, text_embeddings):
    optimizer.zero_grad()
    loss = cfm.compute_loss(model, images, text_embeddings)
    loss.backward()
    optimizer.step()
    return loss.item()


def main():

    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() and config['device'] == 'cuda' else "cpu"
    print(f"Using device: {device}")
    
    
    save_dir = config['training'].get('save_dir', 'outputs')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Model checkpoints will be saved to: {save_dir}")

    # --- モデルとオプティマイザの準備 ---
    condition_model = ConditioningModel(config).to(device)
    print(f"Loaded ConditioningModel (Mode: {condition_model.condition_type})")
    model = TextConditionedUNet(config).to(device)
    cfm = OTConditionalFlowMatching(sigma_min=1e-5)
    params_to_optimize = list(model.parameters()) + list(condition_model.parameters())
    optimizer = optim.Adam(params_to_optimize, lr=config['training']['lr'])
    

    # --- データローダーの準備 ---
    img_channels = config['data']['channels']
    mean = [0.5] * img_channels
    std = [0.5] * img_channels
    print(f"Data channels: {img_channels}, Using Normalize(mean={mean}, std={std})")

    h = config['data']['height']
    w = config['data']['width']

    image_transform = transforms.Compose([
        transforms.Resize((h, w)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean, std)
    ])
    
    dataset = T2IDataset(
        root_dir=config['data']['root_dir'], 
        metadata_file=config['data']['metadata_file'], 
        image_transform=image_transform,
        target_channels=img_channels 
    )
    
    collate_fn = T2ICollate()

    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['training'].get('num_workers', 4) 
    )

    # --- 学習ループ ---
    num_epochs = config['training'].get('num_epochs', 10) 
    print(f"Training Step Start (for {num_epochs} epochs)...")
    
    model.train()
    condition_model.train()

    global_step = 0
    for epoch in range(num_epochs):
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        total_loss = 0.0
        steps_in_epoch = 0

        for batch in pbar:
            try:
                images = batch["pixel_values"].to(device)
                text_embeddings = condition_model(batch, device)
                
                loss = train_step(cfm, model, optimizer, images, text_embeddings)
                
                total_loss += loss
                steps_in_epoch += 1
                global_step += 1

                pbar.set_postfix({"loss": f"{loss:.4f}", "avg_loss": f"{total_loss / steps_in_epoch:.4f}"})

            except Exception as e:
                print(f"Error during training step {global_step}: {e}")
                continue 
        
        avg_epoch_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

        # --- ★ エポックごとのモデル保存 ---
        epoch_save_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'condition_model_state_dict': condition_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # オプティマイザの状態も保存推奨
            'loss': avg_epoch_loss,
        }, epoch_save_path)
        print(f"Saved checkpoint to {epoch_save_path}")

    # --- 学習ループ終了 ---
    
    print(f"Training finished after {global_step} steps.")
    
    # 最終モデルとして、別途 "model_final.pth" という名前でも保存しておくと便利です
    final_save_path = os.path.join(save_dir, "model_final.pth")
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'condition_model_state_dict': condition_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_epoch_loss,
    }, final_save_path)
    
    print(f"Final model saved to {final_save_path}")

if __name__ == "__main__":
    main()
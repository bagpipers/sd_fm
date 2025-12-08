import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class T2IDataset(Dataset):
    """
  
    """
    def __init__(self, root_dir, metadata_file="metadata.csv", 
                 image_transform=None, 
                 target_channels: int = 3):
        
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.image_transform = image_transform
        self.target_channels = target_channels
        if self.target_channels == 1:
            self.convert_mode = 'L'
        elif self.target_channels == 3:
            self.convert_mode = 'RGB'
        elif self.target_channels == 4:
            self.convert_mode = 'RGBA'
        else:
            raise ValueError(f"Unsupported target_channels: {target_channels}. Must be 1, 3, or 4.")
            
        metadata_path = os.path.join(root_dir, metadata_file)
        file_extension = os.path.splitext(metadata_path)[1].lower()
        try:
            if file_extension == ".csv":
                self.metadata = pd.read_csv(metadata_path)
            elif file_extension == ".jsonl":
                self.metadata = pd.read_json(metadata_path, lines=True)
            else:
                raise ValueError(f"Unsupported metadata file format: {file_extension}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            
        if "image_file" not in self.metadata.columns:
            raise ValueError("metadataに 'image_file' カラムがありません。")
        if "positive_prompt" not in self.metadata.columns:
            raise ValueError("metadataに 'positive_prompt' カラムがありません。")
        if "negative_prompt" not in self.metadata.columns:
            print("Warning: 'negative_prompt' column not found. Using empty strings.")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.loc[idx, "image_file"]
        img_path = os.path.join(self.image_dir, img_name)
        try:
            image = Image.open(img_path)
        except FileNotFoundError:
            print(f"Warning: Image file not found {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self)) 
        image = image.convert(self.convert_mode)

        if self.image_transform:
            image = self.image_transform(image)
            
        positive_prompt = self.metadata.loc[idx, "positive_prompt"]
        
        if "negative_prompt" in self.metadata.columns:
            negative_prompt = self.metadata.loc[idx, "negative_prompt"]
        else:
            negative_prompt = "" 
            
        return {
            "image": image,
            "positive_prompt": str(positive_prompt), 
            "negative_prompt": str(negative_prompt)
        }



class T2ICollate:
    """
    
    バッチを辞書型にまとめるだけのシンプルなクラス
    """
    def __init__(self):
        pass

    def __call__(self, batch):
        images = []
        pos_prompts = []
        neg_prompts = []
        
        for item in batch:
            images.append(item["image"])
            pos_prompts.append(item["positive_prompt"])
            neg_prompts.append(item["negative_prompt"])
        image_batch = torch.stack(images)
        return {
            "pixel_values": image_batch,
            "positive_prompt": pos_prompts, # (★ 生の文字列リスト)
            "negative_prompt": neg_prompts, # (★ 生の文字列リスト)
        }
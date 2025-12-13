import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class T2IDataset(Dataset):
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
            raise ValueError(f"Unsupported target_channels: {target_channels}")
            
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
            
        
        self.has_class_id = "class_id" in self.metadata.columns

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.loc[idx, "image_file"]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path)
            image.load() 
            image = image.convert(self.convert_mode)
            positive_prompt = self.metadata.loc[idx, "positive_prompt"]
            if "negative_prompt" in self.metadata.columns:
                negative_prompt = self.metadata.loc[idx, "negative_prompt"]
            else:
                negative_prompt = ""

        except (FileNotFoundError, OSError, IOError, ValueError) as e:
            dummy_size = (256, 256) 
            if self.target_channels == 1:
                image = Image.new("L", dummy_size, color=255) 
            else:
                image = Image.new("RGB", dummy_size, color=(255, 255, 255)) 
            positive_prompt = "error_placeholder_image" 
            negative_prompt = "quality_image"
        
        if self.image_transform:
            try:
                image = self.image_transform(image)
            except Exception as e:
                print(f"Critical Transform Error at {idx}: {e}")
                raise e

        
        data_item = {
            "image": image,
            "positive_prompt": str(positive_prompt), 
            "negative_prompt": str(negative_prompt)
        }

        if self.has_class_id:
            
            class_id_val = self.metadata.loc[idx, "class_id"]
            
            if pd.isna(class_id_val):
              
                data_item["class_id"] = 0 
            else:
                data_item["class_id"] = int(class_id_val)

        return data_item
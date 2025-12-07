import os
import argparse
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import pandas as pd

def prepare_imagenet(args):
    # 1. 保存先ディレクトリの作成
    root_dir = args.output_dir
    image_dir = os.path.join(root_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    print(f"Downloading/Loading ImageNet-1k...")
    print(f"Cache directory: {args.cache_dir}")

    # 2. データセットのロード
    # split="train" を指定。streaming=Trueにするとディスク容量を節約できますが、
    # 学習速度を上げたい場合はダウンロード(streaming=False)推奨です。
    # ここでは確実にローカルに落とすため streaming=False (デフォルト) を想定しますが、
    # cache_dir を指定することで保存場所を制御します。
    dataset = load_dataset(
        "ILSVRC/imagenet-1k", 
        split="train", 
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )

    # テスト用にサンプル数を制限する場合
    if args.max_samples is not None:
        print(f"Limiting to {args.max_samples} samples for testing.")
        dataset = dataset.select(range(args.max_samples))

    metadata = []
    
    # 3. データの処理
    print("Processing images and creating metadata...")
    for i, item in enumerate(tqdm(dataset)):
        # item['image']: PIL Image
        # item['label']: int (0-999)
        
        image = item['image']
        label_id = item['label']
        
        # ラベルIDからクラス名（テキスト）を取得
        # datasetsの機能を使って int -> str 変換
        label_name = dataset.features['label'].int2str(label_id)
        
        # ファイル名を決定 (例: 0000000.jpg)
        image_filename = f"{i:07d}.jpg"
        save_path = os.path.join(image_dir, image_filename)
        
        # 画像保存 (RGB変換して保存)
        # ImageNetには稀にグレースケールが混じっているためRGB化必須
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        image.save(save_path, quality=95)

        # メタデータ作成
        # use_class_name_as_prompt: 
        #   True  -> prompt="goldfish" (CLIP等のテキスト条件付け用)
        #   False -> prompt="0" (Class ID条件付け用)
        if args.use_class_name:
            positive_prompt = label_name # "tench, Tinca tinca"
        else:
            positive_prompt = str(label_id) # "0"

        metadata.append({
            "image_file": image_filename,
            "positive_prompt": positive_prompt,
            "negative_prompt": "",
            "class_id": label_id,     # 将来的にIDが必要な場合のために保存
            "class_name": label_name  # 将来的に名前が必要な場合のために保存
        })

    # 4. メタデータの保存
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(root_dir, "metadata.csv")
    df.to_csv(csv_path, index=False)

    print(f"Done. Processed {len(df)} images.")
    print(f"Data saved to: {root_dir}")
    print(f"Metadata saved to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare ImageNet dataset.")
    
    # 必須: 出力先ディレクトリ
    parser.add_argument("--output_dir", type=str, default="/mnt/k/2025/imagenet_dataset", 
                        help="Where to save the processed images and metadata.")
    
    # 重要: ダウンロードキャッシュ先 (大容量ディスクを指定してください)
    parser.add_argument("--cache_dir", type=str, default="/mnt/k/huggingface_cache", 
                        help="Directory for Hugging Face download cache.")
    
    # オプション: クラス名を使うかIDを使うか
    parser.add_argument("--use_class_name", action="store_true", 
                        help="If True, uses class name (e.g. 'goldfish') as positive_prompt. If False, uses ID (e.g. '1').")
    
    # オプション: テスト用に数を制限
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Limit the number of images to process (for debugging).")

    args = parser.parse_args()
    prepare_imagenet(args)
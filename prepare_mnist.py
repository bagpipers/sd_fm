import os
import torchvision
from PIL import Image
import pandas as pd
from tqdm import tqdm

# 保存先ディレクトリを作成
root_dir = "/mnt/k/2025/mnist_dataset"
image_dir = os.path.join(root_dir, "images")
os.makedirs(image_dir, exist_ok=True)

print("Downloading and processing MNIST...")
# MNISTデータセットをダウンロード (train=Trueで訓練データを取得)
mnist_data = torchvision.datasets.MNIST(root="./data", train=True, download=True)

metadata = [] # メタデータを格納するリスト

# データを（画像ファイル, キャプション）の形に変換
for i, (image, label) in enumerate(tqdm(mnist_data)):
    # imageは (PIL Image)
    # labelは (int) 0-9

    # 1. 画像を保存
    #    ファイル名を 000000.pngのようにゼロパディングすると管理しやすい
    image_filename = f"{i:06d}.png"
    image_path = os.path.join(image_dir, image_filename)
    
    # MNISTはグレースケールなのでRGBに変換（将来のT2Iモデルは大抵RGBを期待するため）
    image.convert("L").save(image_path)

    # 2. メタデータを作成
    #    ご要望通り、ラベル(int)を文字列の"positive_prompt"とします
    #    "negative_prompt"は空文字列にします
    metadata.append({
        "image_file": image_filename,
        "positive_prompt": str(label),
        "negative_prompt": "" # negative_prompt用のカラムも用意しておく
    })

# メタデータをCSVファイルとして保存
df = pd.DataFrame(metadata)
df.to_csv(os.path.join(root_dir, "metadata.csv"), index=False)

print(f"Done. Prepared {len(df)} images in {image_dir}")
print(f"Metadata saved to {os.path.join(root_dir, 'metadata.csv')}")
import os
import shutil
import random
import glob
from pathlib import Path
from tqdm import tqdm

# Configuration
SOURCE_DIRS = [
    "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/train",
    "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset/val"
]
DEST_DIR = "/Users/mtejeshx37/Analysis-of-rice-pad/Rice_Dataset_Split"
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # Train, Val, Test
SEED = 42

def split_dataset():
    random.seed(SEED)
    
    # 1. Identify all classes from the first source dir (assuming consistency)
    classes = [d.name for d in Path(SOURCE_DIRS[0]).iterdir() if d.is_dir()]
    classes.sort()
    
    print(f"Found {len(classes)} classes: {classes}")
    
    # Create destination directories
    for split in ['train', 'val', 'test']:
        for cls in classes:
            os.makedirs(os.path.join(DEST_DIR, split, cls), exist_ok=True)
            
    # 2. Iterate per class to preserve balance
    for cls in tqdm(classes, desc="Processing Classes"):
        all_images = []
        
        # Aggregate images from all source directories
        for source_dir in SOURCE_DIRS:
            cls_path = Path(source_dir) / cls
            if cls_path.exists():
                # Get all image files
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
                for ext in extensions:
                    all_images.extend(list(cls_path.glob(ext)))
        
        # Remove duplicates if any (based on filename) to prevent leakage
        # This assumes unique filenames across the dataset. 
        # If filenames are generic (e.g., 001.jpg in both train/val but different img), 
        # this might be risky, but usually datasets have unique IDs.
        # Let's check by filename.
        unique_images = {}
        for img_path in all_images:
            if img_path.name not in unique_images:
                unique_images[img_path.name] = img_path
        
        images_list = list(unique_images.values())
        random.shuffle(images_list)
        
        total_images = len(images_list)
        train_count = int(total_images * SPLIT_RATIOS[0])
        val_count = int(total_images * SPLIT_RATIOS[1])
        # Test gets the rest to ensure sum is total
        
        train_imgs = images_list[:train_count]
        val_imgs = images_list[train_count:train_count + val_count]
        test_imgs = images_list[train_count + val_count:]
        
        # Helper to copy
        def copy_files(files, split_name):
            for f in files:
                shutil.copy2(f, os.path.join(DEST_DIR, split_name, cls, f.name))
                
        copy_files(train_imgs, 'train')
        copy_files(val_imgs, 'val')
        copy_files(test_imgs, 'test')
        
        # print(f"Class {cls}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\nDataset split complete!")
    print(f"Output directory: {DEST_DIR}")
    
    # Validation step: Count files
    print("\nVerifying counts...")
    for split in ['train', 'val', 'test']:
        count = len(list(Path(DEST_DIR).glob(f'{split}/*/*')))
        print(f"{split.upper()}: {count} images")

if __name__ == "__main__":
    split_dataset()

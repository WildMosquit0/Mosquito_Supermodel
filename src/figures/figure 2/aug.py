import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
import shutil

def augment_image(image_path, output_dir, scale=0.5, degrees=45, shear=0.1, mosaic=1.0, hsv_h=0.015, hsv_s=0.7, translate=0.1, fliplr=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image: {image_path}")
        return
    iamge_name = os.path.basename(image_path)
    height, width = image.shape[:2]
    
    transforms = {
        "scaled": A.Affine(scale=(1 - scale, 1 + scale), p=1.0),
        "rotated": A.Affine(rotate=(-degrees, degrees), p=1.0),
        "sheared": A.Affine(shear=shear * 100, p=1.0),
        "hsv_adjusted": A.HueSaturationValue(hue_shift_limit=int(hsv_h * 255), sat_shift_limit=int(hsv_s * 255), p=1.0),
        "translated": A.ShiftScaleRotate(shift_limit=translate, scale_limit=0, rotate_limit=0, p=1.0),
        "flipped_lr": A.HorizontalFlip(p=fliplr)
    }
    
    for i, (name, transform) in enumerate(transforms.items()):
        augmented = transform(image=image)['image']
        output_path = os.path.join(output_dir, f"{iamge_name}_{name}.jpg")
        cv2.imwrite(output_path, augmented)

def process_directory(input_path, output_path):
    if os.path.isfile(input_path):
        augment_image(input_path, output_path)
    elif os.path.isdir(input_path):
        for img_name in tqdm(os.listdir(input_path)):
            img_path = os.path.join(input_path, img_name)
            augment_image(img_path, output_path)
    else:
        print("Invalid input path.")

if __name__ == "__main__":
    input_path = "/home/bohbot/workspace/images/supermodel/figure 2"
    output_path = "/home/bohbot/workspace/images/supermodel/figure 2"
    process_directory(input_path, output_path)
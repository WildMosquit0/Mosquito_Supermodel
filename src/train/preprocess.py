import os
import glob
from random import randint
from PIL import Image, UnidentifiedImageError
from multiprocessing import Pool, cpu_count

class CropImages:
    def __init__(self, images_dir, annotations_dir, crop_size, num_crops):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.image_paths = glob.glob(os.path.join(self.images_dir, '*'))

    def crop_single_image(self, image_path):
        try:
            image = Image.open(image_path)
            width, height = image.size
            crops = []

            for _ in range(self.num_crops):
                left = randint(0, width - self.crop_size[0])
                top = randint(0, height - self.crop_size[1])
                right = left + self.crop_size[0]
                bottom = top + self.crop_size[1]
                crops.append(image.crop((left, top, right, bottom)))

            output_dir = os.path.join(self.images_dir, 'crops')
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.basename(image_path).split('.')[0]

            for i, crop in enumerate(crops):
                crop.save(os.path.join(output_dir, f"{base_name}_crop_{i}.png"))
        except (UnidentifiedImageError, OSError):
            pass

    def process_all_images(self):
        with Pool(cpu_count()-2) as pool:
            pool.map(self.crop_single_image, self.image_paths)

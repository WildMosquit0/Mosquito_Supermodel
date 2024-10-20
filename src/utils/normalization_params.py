import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class FlatImageDataset(Dataset):
    """
    Custom dataset for images in a flat directory (no class subfolders).
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image

def calculate_normalization_params(image_dir, batch_size=32, resize_size=(640, 640)):
    """
    Calculate the mean and standard deviation of all images in a given flat directory.

    Args:
        image_dir (str): Directory containing the images.
        batch_size (int): Batch size for loading the images. Default is 32.
        resize_size (tuple): Size to resize the images. Default is (640, 640).

    Returns:
        tuple: A tuple containing (mean, std) for each channel (R, G, B).
    """
    # Define a transform that resizes and converts the images to tensors
    transform = transforms.Compose([
        transforms.Resize(resize_size),  # Resize all images to the same size
        transforms.ToTensor()            # Convert to tensor
    ])

    # Load the dataset using the FlatImageDataset
    dataset = FlatImageDataset(image_dir, transform=transform)
    
    # Use DataLoader to load the images in batches
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Variables to store sum and sum of squares for each channel
    channel_sum = torch.zeros(3)
    channel_sum_squared = torch.zeros(3)
    total_pixels = 0

    print("Calculating mean and standard deviation...")
    
    # Iterate through the DataLoader
    for images in tqdm(dataloader):
        # Number of pixels in each image
        num_pixels_per_batch = images.size(0) * images.size(2) * images.size(3)  # Batch size * height * width

        # Sum of pixel values for each channel
        channel_sum += torch.sum(images, dim=[0, 2, 3])
        
        # Sum of squared pixel values for each channel
        channel_sum_squared += torch.sum(images ** 2, dim=[0, 2, 3])
        
        # Update the total number of pixels
        total_pixels += num_pixels_per_batch

    # Calculate mean for each channel
    mean = channel_sum / total_pixels

    # Calculate variance for each channel: E[X^2] - (E[X])^2
    variance = (channel_sum_squared / total_pixels) - (mean ** 2)

    # Calculate standard deviation (sqrt of variance)
    std = torch.sqrt(variance)

    return mean.tolist(), std.tolist()

if __name__ == '__main__':
    image_dir = '/home/ziv/workspace/Mosquito_Supermodel/benchmark/Images'
    mean, std = calculate_normalization_params(image_dir)
    print("Mean:", mean)
    print("Standard deviation:", std)
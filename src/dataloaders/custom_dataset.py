from torch.utils.data import Dataset
from PIL import Image

class ImagePathDataset(Dataset):
    """
    Custom dataset to load images based on a list of file paths provided in a .txt file.
    """
    def __init__(self, txt_file, transform=None):
        """
        Args:
            txt_file (str): Path to the .txt file with image paths.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        with open(txt_file, 'r') as file:
            self.image_paths = file.readlines()
        self.image_paths = [x.strip() for x in self.image_paths]  # Remove any extra spaces or newlines
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image

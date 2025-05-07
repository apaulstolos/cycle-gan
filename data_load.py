import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageTransform(Dataset):
    def __init__(self, root_photo, root_nature):
        self.transform = transforms.Compose([
            transforms.Resize((286, 286)),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        
        self.photo_paths = sorted([os.path.join(root_photo, f) for f in os.listdir(root_photo) if f.endswith(('jpg', 'png'))])
        self.nature_paths = sorted([os.path.join(root_nature, f) for f in os.listdir(root_nature) if f.endswith(('jpg', 'png'))])

    def __len__(self):
        return len(self.photo_paths)

    def __getitem__(self, index):
        photo = Image.open(self.photo_paths[index]).convert("RGB")
        nature = Image.open(self.nature_paths[index]).convert("RGB")

        seed = random.random()
        random.seed(seed)
        photo = self.transform(photo)
        random.seed(seed)
        nature = self.transform(nature)

        return {"photo": photo, "nature": nature}

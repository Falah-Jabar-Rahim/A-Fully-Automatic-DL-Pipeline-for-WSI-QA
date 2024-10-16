import os
import random
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import ndimage
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import KFold
from torchvision import transforms as T
from torchvision.transforms import functional as F
from collections import Counter
from skimage.transform import resize
from PIL import Image

def transform_convert(img_tensor, transform):
    """
    param img_tensor: tensor
    param transforms: torchvision.transforms
    """
    if 'Normalize' in str(transform):
        normal_transform = list(filter(lambda x: isinstance(x, T.Normalize), transform.transforms))
        mean = torch.tensor(normal_transform[0].mean, dtype=img_tensor.dtype, device=img_tensor.device)
        std = torch.tensor(normal_transform[0].std, dtype=img_tensor.dtype, device=img_tensor.device)
        img_tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
 
    # C x H x W  ---> H x W x C
    img_tensor = img_tensor.transpose(0, 2).transpose(0, 1)
 
    if 'ToTensor' in str(transform) or img_tensor.max() < 1:
        img_tensor = img_tensor.detach().numpy() * 255
 
    if isinstance(img_tensor, torch.Tensor):
        img_tensor = img_tensor.numpy()
 
    if img_tensor.shape[2] == 3:
        img = Image.fromarray(img_tensor.astype('uint8')).convert('RGB')
    elif img_tensor.shape[2] == 1:
        img = Image.fromarray(img_tensor.astype('uint8')).squeeze()
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_tensor.shape[2]))

class RandomVerticalFlip(object):
    def __init__(self, vertical_prob):
        self.vertical_prob = vertical_prob
 
    def __call__(self, image, mask=None):
        if random.random() < self.vertical_prob:
            image = F.vflip(image) 
            if mask is not None:
                mask = F.vflip(mask)
        return image, mask

class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
 
    def __call__(self, image, mask=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if mask is not None:
                mask = F.hflip(mask)
        return image, mask

class RandomRotation(object):
    def __init__(self, rotate_prob, max_angle=90):
        self.rotate_prob = rotate_prob
        self.max_angle = max_angle

    def __call__(self, image, mask=None):
        if random.random() < self.rotate_prob:
            angle = np.random.randint(-self.max_angle, self.max_angle)
            image = image.rotate(angle)
            if mask is not None:
                mask = mask.rotate(angle)
        return image, mask

class ColorJitter(object):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, image, target):
        image = T.ColorJitter(brightness=self.brightness,
                              contrast=self.contrast,
                              saturation=self.saturation,
                              hue=self.hue)(image)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, mask=None):
        image = F.normalize(image, mean=self.mean, std=self.std)       
        return image, mask

class ToTensor(object):
    def __call__(self, image, mask=None):
        image = F.to_tensor(image)
        if mask is not None:

            # mask = torch.from_numpy(np.asarray(mask))
            # if warning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. This means you can write to the underlying (supposedly non-writeable) NumPy array using the tensor
            mask = torch.from_numpy(np.asarray(np.copy(mask)))
        return image, mask

class resize_img_msk(object):
    def __init__(self, img_size):
        self.img_size = img_size
    def __call__(self, image, mask=None):
        image = image.resize((self.img_size, self.img_size), Image.NEAREST)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
        return image, mask

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class DHUnet_dataset(Dataset):
    def __init__(self, list_dir, split, fold_no=0, total_fold=5, img_size=None):
        self.split = split 
        self.list_dir = list_dir
        self.img_size = img_size
        random.seed(301)
        np.random.seed(301)
        torch.manual_seed(301)
        torch.cuda.manual_seed(301)

        if self.split == "test":
            self.image_list = self.get_sample_list("test_images")
            self.mask_list = self.get_sample_list("test_masks")
        else: # train val 
            self.image_list = self.get_sample_list("train_images")
            self.mask_list = self.get_sample_list("train_masks")

            # Take the fold_no th fold data according to cross-validation
            if fold_no != -1: # Use all data to train
                kfold = KFold(n_splits=total_fold, shuffle=True)
                for i, (train_index, val_index) in enumerate(kfold.split(self.image_list, self.mask_list)):
                    if i == fold_no:
                        print((train_index), (val_index))
                        if self.split == "train":
                            self.image_list = np.array(self.image_list)[train_index]
                            self.mask_list = np.array(self.mask_list)[train_index]
                            print(self.image_list[100], self.mask_list[100])
                        if self.split == "val":
                            self.image_list = np.array(self.image_list)[val_index]
                            self.mask_list = np.array(self.mask_list)[val_index]
                            print(self.image_list[100], self.mask_list[100])
                        break
            elif fold_no == -1 and self.split == 'val': # Use all data training, the remaining validation set is empty
                self.image_list = []
                self.mask_list = []
                
        if split == "train":
            self.img_transform = Compose([
                                    ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                resize_img_msk(self.img_size),
                RandomHorizontalFlip(flip_prob=0.5),
                                    RandomVerticalFlip(vertical_prob=0.5),
                                    #RandomRotation(rotate_prob=0.5),
                                    ToTensor(),
                                    #Normalize([0.485, 0.456, 0.406],
                                    #          [0.229, 0.224, 0.225])
                                    ])
        else: # val test
            self.img_transform = Compose([
                resize_img_msk(self.img_size),
                ToTensor(),
                #Normalize([0.485, 0.456, 0.406],
                #           [0.229, 0.224, 0.225])
                ])
    def __len__(self):
        return len(self.image_list) 

    def get_sample_list(self, sample_name):
        file_lines = open(os.path.join(self.list_dir, sample_name + '.txt')).readlines()
        sample_list = []
        for line in file_lines: 
            line = line.strip('\n') 
            if line:
                sample_list.append(line)
        return sample_list

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __getitem__(self, idx):
        assert os.path.basename(self.image_list[idx]).split('.')[0] == os.path.basename(self.mask_list[idx]).split('.')[0]
        image = self.rgb_loader(self.image_list[idx])
        mask = self.binary_loader(self.mask_list[idx])
        image, mask = self.img_transform(image, mask)
        sample = {'image':image, 'mask':mask}
        sample['case_name'] = self.image_list[idx].split('/',1)[-1]
        return sample

if __name__ == '__main__':

    from PIL import Image
    from collections import Counter
    tt = DHUnet_dataset(list_dir='/home/falah/Desktop/WorkingFolder/Artifact_detection/DHUnet-master/my_dataset',split='train')
    # tt = DHUnet_dataset(list_dir='/home/wl/lian/Medical_Image/DHUnet/lists/lists_Liver',split='train')
    # tt = DHUnet_dataset(list_dir='/home/wl/lian/Medical_Image/DHUnet/lists/lists_Liver',split='train')

    import os
    import numpy as np
    from PIL import Image

    # Define the directory to save the images
    save_dir = 'visualization/'
    os.makedirs(save_dir, exist_ok=True)

    for i in range(2, 3):
        sample = tt.__getitem__(i)
        image, mask, name = sample['image'], sample['mask'], sample['case_name']

        # Convert image and mask to NumPy arrays
        image_np = np.transpose(image.numpy(), (1, 2, 0))
        image_np = (image_np * 255).astype(np.uint8)

        mask_np = np.array(mask)

        # Create the directory if it doesn't exist

        # Save the images
        name_save = name.split('/')[-1]
        image_filename = os.path.join(save_dir, name_save)
        mask_filename = os.path.join(save_dir, f'{i}_{name_save}_mask.png')
        Image.fromarray(image_np).save(image_filename)
        Image.fromarray(mask_np).save(mask_filename)

    # print(tt.__len__())
    #
    # for i in range(1, 10):
    #     sample = tt.__getitem__(i)
    #     image, mask, name = sample['image'],sample['mask'],sample['case_name']
    #     os.makedirs('visualization/Liver_dataset',exist_ok=True)
    #
    #     print(Counter(np.array(mask).flatten()))
    #     palette=[]
    #     for j in range(256):
    #         palette.extend((j,j,j))
    #         palette[:3*6]=np.array([
    #                             [0, 0, 0],
    #                             [0,255,0],
    #                             [0,0,255],
    #                             [255,255,0],
    #                             [255,0,0],
    #                          ], dtype='uint8').flatten()
    #     mask = mask.convert('P')
    #     mask.putpalette(palette)
    #     filename = os.path.basename(name).split('.')[0]
    #     image.convert('RGB').save('visualization/Liver_dataset/'+str(i) + '_' + filename + ".jpg")
    #     mask.save('visualization/Liver_dataset/' + str(i) + '_' + filename + "_mask.png")

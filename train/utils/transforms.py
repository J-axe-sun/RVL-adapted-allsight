from torchvision import transforms
import torch
from PIL import ImageFilter
import random

# display visual model inputs
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def get_transforms(im_size):
    """
    Get transformations for training, augmentation, and testing images.

    Args:
        im_size (int): Size of the input images after resizing.

    Returns:
        tuple: A tuple containing three torchvision transforms:
               - train_transform: Transformations for training images.
               - aug_transform: Augmentation transformations.
               - test_transform: Transformations for testing images.
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.RandomChoice([
            # transforms.RandomRotation(3),  # rotate +/- 10 degrees
            transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
            # transforms.ColorJitter(brightness=1.0, contrast=0.1, saturation=0.5, hue=0.1),
        ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    aug_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.5),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        # transforms.RandomChoice([
        #     transforms.RandomRotation(3),  # rotate +/- 10 degrees
        #     transforms.RandomAffine(degrees=0, translate=(0.02, 0.02))
        # ]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, aug_transform, test_transform


import torchvision.transforms.functional as TF

class ToGrayscale(object):
    """
    Convert image to grayscale version of image.

    Args:
        num_output_channels (int, optional): Number of channels for the output image. Default is 1.
    """

    def __init__(self, num_output_channels=1):
        self.num_output_channels = num_output_channels

    def __call__(self, img):
        """
        Apply grayscale conversion to the input image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Grayscale version of the input image.
        """
        return TF.to_grayscale(img, self.num_output_channels)


class AdjustGamma(object):
    """
    Perform gamma correction on an image.

    Args:
        gamma (float): Non-negative real number, gamma value for correction.
        gain (float, optional): Multiplicative factor. Default is 1.
    """

    def __init__(self, gamma, gain=1):
        self.gamma = gamma
        self.gain = gain

    def __call__(self, img):
        """
        Apply gamma correction to the input image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image: Image after gamma correction.
        """
        return TF.adjust_gamma(img, self.gamma, self.gain)



class GaussianBlur(object):
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709

    Args:
        sigma (list or tuple): Range [min_sigma, max_sigma] for blur radius.

    """

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        """
        Apply Gaussian blur to the input image.

        Args:
            x (PIL.Image): Input image.

        Returns:
            PIL.Image: Image after Gaussian blur.
        """
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class TwoCropsTransform:
    """
    Take two random crops of one image as the query and key.

    Args:
        base_transform (callable): Transformations to apply to the image before cropping.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        """
        Apply two random crops to the input image.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Concatenation of two cropped versions of the input image.
        """
        q = self.base_transform(x)
        k = self.base_transform(x)
        return torch.cat([q, k], dim=0)
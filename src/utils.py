import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt

def load_data(batch_size = 32,data='mnist'):
    '''
    Input arguments: batch size , dataset name
    returns: train_loader, test_loader
    '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5]), ])

    if data == 'fashion':
        # FashionMNIST Dataset
        train_dataset = datasets.FashionMNIST(root='data', train=True, download=True, transform=transform)
        test_dataset  = datasets.FashionMNIST(root='data', train=False, download=True, transform=transform)

    elif data == 'mnist': 
        #  MNIST Dataset
        train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
        test_dataset  = datasets.MNIST(root='data', train=False, download=True, transform=transform)
    else:
        raise ValueError("Error: Data must be 'mnist' or 'fashion' only.")
    # Data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,test_loader


def add_noise(imgs, noise_factor=0.5):
    noisy_imgs = imgs + noise_factor * torch.randn_like(imgs)
    return torch.clip(noisy_imgs, 0., 1.)

def show_images(denoised_imgs,noisy_test_imgs, test_imgs ,n=8):
    
    plt.figure(figsize=(18,6))
    for i in range(n):
        # Noisy
        ax = plt.subplot(3, n, i+1)
        plt.imshow(noisy_test_imgs[i].cpu().squeeze(), cmap='gray')
        plt.title("Noisy")
        plt.axis('off')
        # Original
        ax = plt.subplot(3, n, i+1+n)
        plt.imshow(test_imgs[i].cpu().squeeze(), cmap='gray')
        plt.title("Original")
        plt.axis('off')
        # Denoised
        ax = plt.subplot(3, n, i+1+2*n)
        plt.imshow(denoised_imgs[i].cpu().squeeze(), cmap='gray')
        plt.title("Denoised")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

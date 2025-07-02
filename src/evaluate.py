import torch

def eval(model,test_loader,device, noise_factor= 0.5):
    from .utils import add_noise

    model.eval()
    test_iter = iter(test_loader)
    test_imgs, _ = next(test_iter)
    test_imgs = test_imgs.to(device)
    noisy_test_imgs = add_noise(test_imgs, noise_factor)
    with torch.no_grad():
        denoised_imgs = model(noisy_test_imgs)
    return denoised_imgs,noisy_test_imgs,test_imgs

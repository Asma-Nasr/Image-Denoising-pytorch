import torch
import torch.nn as nn
import torch.optim as optim
from .utils import add_noise
import os

def train(model,train_loader,device,epochs, learning_rate = 1e-3,noise_factor=0.5):
    '''
    Train the model 
    Inputs: 
    learning rate,
    epochs: number of epochs
    noise_factor
    returns the trained mddel 
    '''
    directory = 'Saved Model'

    # Creating the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    
    best_loss = float('inf')
    print('--------------------Training Started---------------------')

    for epoch in range(epochs):

        model.train()
        train_loss = 0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            noisy_imgs = add_noise(imgs, noise_factor)
            outputs = model(noisy_imgs)
            loss = criterion(outputs, imgs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")
        if train_loss < best_loss:
            best_loss = train_loss
        best_state_dict = model.state_dict()
    print('--------------------Training Ended ----------------------')
    torch.save(best_state_dict, os.path.join(directory, 'best_model.pth'))    
    return model

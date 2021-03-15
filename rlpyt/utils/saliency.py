import torch
import matplotlib.pyplot as plt
import numpy as np


def saliency(img, model, save_path):

    # We don't need gradients wrt weights for a trained model
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    input = torch.from_numpy(img).float().unsqueeze(0)

    # Get gradient wrt action
    input.requires_grad = True
    action = model(input, prev_action=None, prev_reward=None)[0]
    action.backward()

    # Get max along channel axis (only using first 3 channels, which are RGB of most recent image)
    slc, _ = torch.max(torch.abs(input.grad[0][:3]), dim=0)
    # Normalize to [0..1]
    slc = (slc - slc.min())/(slc.max()-slc.min())

    # Plot image and its saliency map
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(np.moveaxis(img,0,-1)[:,:,:3])   # most recent image
    plt.subplot(1, 2, 2)
    plt.imshow(slc.numpy(), cmap=plt.cm.hot)
    # plt.show()
    plt.savefig(save_path)
    plt.close()

    for param in model.parameters():
        param.requires_grad = True

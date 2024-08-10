import torch
import matplotlib.pyplot as plt
import numpy as np

## Create a function to load weights from a .ckpt file
def load_ckpt(path):
# Path to your .ckpt file
    # Load the checkpoint
    checkpoint = torch.load(path)

    print("Checkpoint contains the following keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\nState Dictionary Contents:")
        for param_tensor in state_dict:
            print(f"{state_dict[param_tensor].size()}\t\t\t\t{param_tensor}")
    else:
        print("\nNo 'state_dict' found in the checkpoint.")

        print("Assuming the entire checkpoint is a state dictionary.\nContents:")
        for param_tensor in checkpoint:
            print(f"{param_tensor}:\t{checkpoint[param_tensor].size()}")

# Function to display a tensor as an image
def display_image(tensor):
    tensor = tensor.detach().cpu().numpy()
    plt.imshow(np.transpose(tensor, (1, 2, 0)))
    plt.show()

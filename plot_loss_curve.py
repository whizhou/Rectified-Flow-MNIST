import matplotlib.pyplot as plt
import pathlib
from pathlib import Path
import torch

if __name__ == "__main__":
    checkpoint_path = Path(__file__).resolve().parent.joinpath(
        'checkpoints', 'v1.1-cfg', 'Unet_final.pth')
    checkpoint = torch.load(checkpoint_path)
    loss_list = checkpoint['loss_list']

    plt.plot(loss_list)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    # plt.savefig('test')
    save_path = Path(__file__).resolve().parent.joinpath(
        'results', 'loss_curve'
    )
    # plt.savefig(save_path)
    plt.show()

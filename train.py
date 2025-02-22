import torch
import os
import yaml
import pathlib
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F
from model.conditional_unet2d import ConditionalUnet2D
from model.rectified_flow import RectifiedFlow


def train(root_dir: str):
    """
    train Rectified Flow model with config
    Args:
        root_dir - str: 项目根目录, 至少有以下文件或目录
            train.py
            model/
            config/
    """

    cur_path = Path(root_dir)

    config_path = cur_path / 'config' / 'flow_mnist.yaml'
    config = yaml.load(open(config_path, 'rb'), Loader=yaml.FullLoader)
    input_dim =config.get('input_dim', 1)
    base_channels = config.get('base_channels', 16)
    global_cond_embed_dim = config.get('global_cond_embed_dim', 128)
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr_adjust_epoch = config.get('lr_adjust_epoch', 50)
    batch_print = config.get('batch_print', True)
    batch_print_interval = config.get('batch_print_interval', 100)
    checkpoint_save = config.get('checkpoint_save', True)
    checkpoint_save_interval = config.get('checkoint_save_interval', 1)
    save_path = config.get('save_path', None)
    use_cfg = config.get('use_cfg', False)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    
    # Train Rectified Flow model

    # Load dataset
    # Transform PIL to tensor and normalize to [-1, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
        transforms.Normalize(mean=[0], std=[1])
    ])

    data_path = cur_path / 'data'
    dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = ConditionalUnet2D(
        input_dim=input_dim, base_channels=base_channels,
        global_cond_embed_dim=global_cond_embed_dim
    )
    model.to(device)
    
    # Load optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    # learning rate adjust
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_adjust_epoch, gamma=0.1)

    # Load rectified flow
    rf = RectifiedFlow()

    loss_list = []

    # Create folder for save path
    save_path = cur_path.joinpath('checkpoints', save_path)
    save_path.mkdir(exist_ok=True)

    # train epochs
    for epoch in range(epochs):
        for batch, data in enumerate(dataloader):
            x1, y = data

            x0 = torch.randn_like(x1)

            x_t, t, v_target = rf.get_train_turple(z0=x0, z1=x1)

            y = y.to(device)
            x_t = x_t.to(device)
            t = t.to(device)
            v_target = v_target.to(device)

            if (use_cfg):
                x_t = torch.cat([x_t, x_t.clone()], dim=0)
                t = torch.cat([t, t.clone()], dim=0)
                v_target = torch.cat([v_target, v_target.clone()], dim=0)
                y = torch.cat([y, -torch.ones_like(y)], dim=0)
                y.to(device)
            else:
                y = None

            optimizer.zero_grad()

            v_pred = model(sample=x_t, timestep=t, global_cond=y)

            loss = F.mse_loss(v_target, v_pred)

            loss.backward()
            optimizer.step()

            if batch_print and batch % batch_print_interval == 0:
                print(f'epoch {epoch}, batch {batch_print}, loss: {loss.item()}')
            
            loss_list.append(loss.item())
        
        scheduler.step()

        if checkpoint_save and epoch % checkpoint_save_interval == 0:
            print(f'Saving model Unet_{epoch} to {save_path.as_posix()}')
            save_dict = dict(model=model.state_dict(),
                             optimizer=optimizer.state_dict(),
                             epoch=epoch,
                             loss_list=loss_list,
                             input_dim=input_dim,
                             base_channels=base_channels,
                             global_cond_embed_dim=global_cond_embed_dim
                             )
            torch.save(save_dict, save_path.joinpath(f'Unet_{epoch}.pth'))

    print(f'Saving model Unet_final to {save_path.as_posix()}')
    save_dict = dict(model=model.state_dict(),
                     optimizer=optimizer.state_dict(),
                     epoch=epochs,
                     loss_list=loss_list,
                     input_dim=input_dim,
                     base_channels=base_channels,
                     global_cond_embed_dim=global_cond_embed_dim)
    torch.save(save_dict, save_path.joinpath(f'Unet_final.pth'))

if __name__ == "__main__":
    print(torch.cuda.__name__)
    root_dir = Path(__file__).resolve().parent  # 默认 train.py 位于项目的根目录下
    # config_path = cur_path.joinpath("config").joinpath("flow_minist.yaml")
    print(root_dir.as_posix())
    # print(root_dir.absolute())
    # print(root_dir.parent.parent.absolute().parent.parent)
    train(root_dir=root_dir.as_posix())
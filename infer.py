import torch
import cv2
import numpy as np
import pathlib
from pathlib import Path
from model.conditional_unet2d import ConditionalUnet2D
from model.rectified_flow import RectifiedFlow
pathlib.PosixPath

def infer(
        checkpoint_path,
        step=50,
        num_imgs=5,
        y=None,
        cfg_scale=7.0,
        save_path=None,
        save_noise_path=None,
        device='cuda'):
    """Generate imgaes with flow matching
    Args:
        checkpoint_path - str: 模型路径
        step - (int, optional): 采样步数, default-50
        num_imgs - (int, optional): 推理一次生成图片数量, default-5
        y - (torch.Tensor, optional): [B], 条件生成的条件, 为数据标签
        cfg_scale - (float, optional): Classifier-free Guidance 的缩放因子, 值越小条件约束越强, default-7.0
        save_path - ([str, pathlib.PurePath], optional): 图像保存路径, 为 './results/save_path' (str) 或 save_path (pathlib.PurePath), default-None
        save_noise_path (str, options): 保存噪声路径, default-None
        device (str, option): 推理设备, default-'cuda'
    """

    if type(save_path) is str:
        root_dir = Path(__file__).resolve().parent
        save_path = root_dir / 'results' / save_path
    save_path.mkdir(exist_ok=True)
    if save_noise_path is not None:
        if type(save_noise_path) is str:
            save_noise_path = Path(__file__).resolve().parent.joinpath('result', 'noise', save_noise_path)
        Path(save_noise_path).mkdir(exist_ok=True)
    
    if y is not None:
        assert len(y.shape) == 1, 'y must be 1D tensor'
        if y.shape[0] == 1:
            y = y.repeat(num_imgs)
        y.to(device)
        
    # Generate images
    # Load model
    checkpoint = torch.load(checkpoint_path)
    model = ConditionalUnet2D(
        checkpoint['input_dim'],
        checkpoint['base_channels'],
        checkpoint['global_cond_embed_dim'],
    )
    model.to(device)
    model.eval()
    model.load_state_dict(checkpoint['model'])

    # Load Rectified Flow
    rf = RectifiedFlow()

    with torch.no_grad():
        for i in range(num_imgs):
            print(f'Generating {i}th image...')

            dt = 1.0 / step

            x_t = torch.randn(1, 1, 28, 28).to(device)
            noise = x_t.detach().cpu().numpy()

            if y is not None:
                y_i = y[i].unsqueeze(0).to(device)
            
            for j in range(step):
                t = j * dt
                t = torch.tensor([t]).to(device)
                if y is not None:
                    v_pred_uncond = model(sample=x_t, timestep=t,
                                          global_cond=-torch.ones(1).to(device))
                    v_pred_cond = model(sample=x_t, timestep=t, global_cond=y_i)
                    v_pred = v_pred_uncond + cfg_scale * (v_pred_cond - v_pred_uncond)
                else:
                    v_pred = model(sample=x_t, timestep=t,
                                   global_cond=-torch.ones(1).to(device))
                x_t = x_t + v_pred * dt
            
            x_t = x_t.clamp(0, 1)
            img = x_t.detach().cpu().numpy()
            img = img[0][0] * 255
            img = img.astype('uint8')
            cv2.imwrite(save_path.joinpath(f'{i}.png'), img)
            if save_noise_path is not None:
                np.save(save_noise_path.joinpath('f{i}.npy'), noise)


if __name__ == "__main__":
    y = []
    for i in range(10):
        y.extend([i] * 10)
    work_dir = Path(__file__).resolve().parent
    print(work_dir)
    infer(
        checkpoint_path=work_dir.joinpath('checkpoints', 'v1.1-cfg', 'Unet_final.pth'),
        step=100,
        num_imgs=100,
        # y=torch.tensor(y),
        cfg_scale=5.0,
        save_path='test'
    )

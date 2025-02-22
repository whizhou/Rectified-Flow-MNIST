import torch
import torch.nn.functional as F

class RectifiedFlow():
    def get_train_turple(self, z0=None, z1=None):
        """
        Args:
            z0: [B,C,H,W], 噪声图像
            z1: [B,C,H,W], 原始图像
        
        Returns:
            z_t: [B,C,H,W], 时间 t 的图像
            t: [B,] 时间 t
            target:
        """
        assert z1 is not None, "offer the origin image"

        t = torch.rand((z1.shape[0], 1, 1, 1))

        if z0 is None:
            z0 = torch.randn_like(z1)

        z_t = t * z1 + (1 - t) * z0
        target = z1 - z0

        return z_t, t.view(-1), target
        

if __name__ == "__main__":
    
    rf = RectifiedFlow()
    x_t, t, target = rf.get_train_turple(torch.randn((2, 3, 4, 4)), torch.ones((2, 3, 4, 4)))
    print(x_t)
    print(t)
    print(target)
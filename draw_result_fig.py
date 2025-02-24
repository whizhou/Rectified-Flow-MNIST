import matplotlib.pyplot as plt
from pathlib import Path
import cv2


def draw_result_fig(label: str):
    img_path = Path(__file__).resolve().parent.joinpath(
        'results', label
    )
    imgs = list(img_path.glob('*.png'))[:100]

    imgs.sort(key=lambda x: int(x.stem))

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))

    for ax, img in zip(axes.flatten(), imgs):
        img = cv2.imread(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.axis('off')
        
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()
    # plt.show()
    save_path = Path(__file__).resolve().parent.joinpath(
        'result_figs', f'{label}.png'
    )
    plt.savefig(save_path)

if __name__ == "__main__":
    draw_result_fig('conditional-v1.2')
    draw_result_fig('unconditional-v1.2')

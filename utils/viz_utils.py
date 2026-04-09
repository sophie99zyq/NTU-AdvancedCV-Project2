import matplotlib.pyplot as plt
import torch


def show_image_grid(images_dict, nrow=8, figsize=(16, 4), title=None, save_path=None):
    n_methods = len(images_dict)
    fig, axes = plt.subplots(n_methods, 1, figsize=(figsize[0], figsize[1] * n_methods))
    if n_methods == 1:
        axes = [axes]
    for ax, (name, imgs) in zip(axes, images_dict.items()):
        imgs = imgs[:nrow].cpu()
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        grid = torch.cat([imgs[i] for i in range(imgs.shape[0])], dim=2)
        ax.imshow(grid.permute(1, 2, 0).numpy().clip(0, 1))
        ax.set_ylabel(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_results_table(results_dict, dataset_name):
    print(f"\n{'='*50}")
    print(f"Results: {dataset_name}")
    print(f"{'='*50}")
    print(f"{'Method':<25} {'Accuracy':>10}")
    print(f"{'-'*35}")
    for method, acc in results_dict.items():
        print(f"{method:<25} {acc:>10.2%}")
    print(f"{'='*50}\n")
    return results_dict

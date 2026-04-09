import os
import shutil


def prepare_cyclegan_data(source_images_dir, target_images_dir, output_dir):
    """Prepare data in CycleGAN's expected format: trainA/, trainB/."""
    for split, src in [('trainA', source_images_dir), ('trainB', target_images_dir)]:
        dst = os.path.join(output_dir, split)
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(src):
            src_path = os.path.join(src, f)
            dst_path = os.path.join(dst, f)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
    print(f"CycleGAN data prepared at {output_dir}")


def get_cyclegan_train_cmd(dataroot, name, n_epochs=100, n_epochs_decay=100,
                            load_size=64, crop_size=64, input_nc=1, output_nc=1,
                            netG='resnet_6blocks', batch_size=4):
    """Generate command to train CycleGAN via !command in Colab."""
    cmd = (
        f"python pytorch-CycleGAN-and-pix2pix/train.py "
        f"--dataroot {dataroot} "
        f"--name {name} "
        f"--model cycle_gan "
        f"--n_epochs {n_epochs} "
        f"--n_epochs_decay {n_epochs_decay} "
        f"--batch_size {batch_size} "
        f"--load_size {load_size} "
        f"--crop_size {crop_size} "
        f"--input_nc {input_nc} "
        f"--output_nc {output_nc} "
        f"--netG {netG} "
        f"--save_epoch_freq 1 "
        f"--no_html"
    )
    return cmd


def get_cyclegan_test_cmd(dataroot, name, load_size=64, crop_size=64,
                           input_nc=1, output_nc=1, netG='resnet_6blocks'):
    """Generate command to run CycleGAN inference."""
    cmd = (
        f"python pytorch-CycleGAN-and-pix2pix/test.py "
        f"--dataroot {dataroot} "
        f"--name {name} "
        f"--model cycle_gan "
        f"--load_size {load_size} "
        f"--crop_size {crop_size} "
        f"--input_nc {input_nc} "
        f"--output_nc {output_nc} "
        f"--netG {netG} "
        f"--no_dropout"
    )
    return cmd

import os
import random
import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

import sys
import numpy as np
import glob
sys.path.append('./')
from TVT.datasets.realesrgan import RealESRGAN_degradation


def parse_args_realsr_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()
    # args for the vsd training
    parser.add_argument("--positive_prompt", type=str, default='')
    parser.add_argument("--negative_prompt", type=str, default='')
    parser.add_argument("--lambda_vsd", default=1.0, type=float)
    parser.add_argument("--lambda_vsd_lora", default=1.0, type=float)
    parser.add_argument("--min_dm_step_ratio", default=0.02, type=float) 
    parser.add_argument("--skiptimestep", default=30, type=int) 
    parser.add_argument("--max_dm_step_ratio", default=0.98, type=float)
    parser.add_argument("--cfg_vsd", default=7.5, type=float)
    parser.add_argument("--snr_gamma_vsd", default=None)
    parser.add_argument("--lora_rank_unet_vsd", default=8, type=int) 
    parser.add_argument("--pretrained_model_name_or_path_vsd", default='', type=str)
    parser.add_argument("--basemodle_path", default='', type=str)
    # args for controlnet
    parser.add_argument("--controlnet_model_path", default='', type=str)
    
    # args for the loss function
    parser.add_argument("--lambda_lpips", default=2, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", default='', type=str)
    parser.add_argument("--testdataset_folder", default='', type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--null_text_ratio", default=1., type=float)

    # validation eval args
    parser.add_argument("--eval_freq", default=500, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=100, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")
    parser.add_argument('--tiled_size', type=int, default=768)
    parser.add_argument('--tiled_overlap', type=int, default=256)


    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path", default='', type=str)
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--upsampler", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=8, type=int) 
    parser.add_argument("--lora_rank_unet2", default=0, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int) 
    parser.add_argument("--time_step", default=999, type=int)
    parser.add_argument("--time_step2", default=250, type=int)
    parser.add_argument("--time_step3", default=999, type=int)
    parser.add_argument("--pretrained_path", default=None, type=str)
    parser.add_argument("--pretrained_unet_path", default=None, type=str) 
    parser.add_argument("--pretrained_vae_path", default=None, type=str)
    parser.add_argument("--stage2", default=None, type=str)
    parser.add_argument("--stage3", default=None, type=str)
    parser.add_argument('--pretrained_clip', type=str, default='', help='path to clip')
    parser.add_argument("--vae4d_path", default='', type=str)


    # training details
    parser.add_argument("--output_dir", default='')
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=123, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps", type=int, default=10_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--use_online_deg", action="store_true",)
    parser.add_argument("--deg_file_path", default="params_pasd.yml", type=str)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')

    # vae lora
    parser.add_argument("--use_vae_encode_lora", action="store_true",)
    parser.add_argument("--use_vae_decode_lora", action="store_true",)

    # use_lr_999noise
    parser.add_argument("--use_lr_999noise", action="store_true",)
    parser.add_argument("--use_lr_concat_lr_999noise", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # # only for debug
    # args.enable_xformers_memory_efficient_attention = True
    # args.use_online_deg = True
    # args.use_lr_concat_lr_999noise = True

    return args


def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T


class PairedSROnlineDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder, split, image_prep, deg_file_path=None, image_size=512, args=None):
        super().__init__()
        self.split = split
        self.args = args
        clip_mean = [0.48145466, 0.4578275, 0.40821073]
        clip_std = [0.26862954, 0.26130258, 0.27577711]
        self.clip_normalize = transforms.Normalize(mean=clip_mean, std=clip_std)

        if split == 'train':
            self.gt_folder = os.path.join(dataset_folder, "gt")
            self.gt_list = []
            self.gt_list += glob.glob(os.path.join(self.gt_folder, '*.png'))

            self.T = build_transform(image_prep)
            self.split = split

            self.degradation = RealESRGAN_degradation(deg_file_path, device='cpu')
            self.crop_preproc = transforms.Compose([
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ])
        elif split == 'test':
            dataset_folder = args.testdataset_folder 
            self.input_folder = os.path.join(dataset_folder, "test_SR_bicubic")
            self.output_folder = os.path.join(dataset_folder, "test_HR")
            
            self.lr_list = []
            self.gt_list = []
            self.lr_list += glob.glob(os.path.join(self.input_folder, '*.png'))
            self.gt_list += glob.glob(os.path.join(self.output_folder, '*.png'))

            self.T = build_transform(image_prep)
            self.split = split
            assert len(self.lr_list) == len(self.gt_list)

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, idx):

        if self.split == 'train':
            gt_img = Image.open(self.gt_list[idx]).convert('RGB')
            gt_img = self.crop_preproc(gt_img)

            output_t, img_t, img_t_noresize = self.degradation.degrade_process(np.asarray(gt_img)/255., resize_bak=True)
            output_t, img_t, img_t_noresize = output_t.squeeze(0), img_t.squeeze(0), img_t_noresize.squeeze(0)

            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            img_t_noresize = F.normalize(img_t_noresize, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            return {
                "output_pixel_values": output_t,
                "conditioning_pixel_values": img_t,
                "conditioning_pixel_values_noresize": img_t_noresize,
                "negative_prompt": self.args.negative_prompt,
            }
            
        elif self.split == 'test':
            input_img = Image.open(self.lr_list[idx]).convert('RGB')
            input_img_noresize = Image.open(self.gt_list[idx].replace('test_HR/','test_LR/')).convert('RGB')
            output_img = Image.open(self.gt_list[idx]).convert('RGB')

            # input images scaled to -1, 1
            img_t = self.T(input_img)
            img_t = F.to_tensor(img_t)

            img_t_noresize = self.T(input_img_noresize)
            img_t_noresize = F.to_tensor(img_t_noresize)
            
            img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
            img_t_noresize = F.normalize(img_t_noresize, mean=[0.5], std=[0.5])
            # output images scaled to -1,1
            output_t = self.T(output_img)
            output_t = F.to_tensor(output_t)
            output_t = F.normalize(output_t, mean=[0.5], std=[0.5])

            return {
                "output_pixel_values": output_t,
                "conditioning_pixel_values": img_t,
                "conditioning_pixel_values_noresize": img_t_noresize,
                "negative_prompt": self.args.negative_prompt,
                "base_name": os.path.basename(self.lr_list[idx]),
            }


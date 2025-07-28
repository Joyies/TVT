import os
import gc
import lpips
import clip
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import copy

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler

import wandb
from cleanfid.fid import get_folder_features, build_feature_extractor, fid_from_feats
import sys
sys.path.append("TVT")
from modelfile.TVTModel import VSD, TVT
from my_utils.training_utils_realsr import parse_args_realsr_training, PairedSROnlineDataset  

from pathlib import Path
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate import DistributedDataParallelKwargs

sys.path.append('TVT')
from TVT.my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
from diffusers.training_utils import compute_snr
import open_clip
from diffusers import DDPMScheduler, AutoencoderKL

from ram.models.ram_lora import ram
from ram import inference_ram as inference


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)

    TVTSR = TVT(args)
    TVTSR.set_train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            TVTSR.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        TVTSR.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # init vsd model
    net_disc = VSD(args=args, accelerator=accelerator)
    net_disc.set_train()

    net_lpips = lpips.LPIPS(net='vgg').cuda()
    net_lpips.requires_grad_(False)

    # # set adapter
    TVTSR.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    # make the optimizer
    layers_to_opt = []
    for n, _p in TVTSR.unet.named_parameters():
        if "lora" in n or 'down_blocks.0' in n or 'up_blocks.3.upsamplers.0' in n or 'up_blocks.4' in n or 'conv_in' in n or 'conv_out' in n:
            assert _p.requires_grad
            layers_to_opt.append(_p) 
    layers_to_opt += list(TVTSR.unet.conv_in.parameters()) 

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    layers_to_opt_disc = []
    for n, _p in net_disc.named_parameters():
        # if _p.requires_grad:
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt_disc.append(_p)
    optimizer_disc = torch.optim.AdamW(layers_to_opt_disc, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
            num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=args.max_train_steps * accelerator.num_processes,
            num_cycles=args.lr_num_cycles, power=args.lr_power)

    # make the dataloader
    dataset_train = PairedSROnlineDataset(dataset_folder=args.dataset_folder, image_prep=args.train_image_prep, split="train", deg_file_path=args.deg_file_path, args=args)
    dataset_val = PairedSROnlineDataset(dataset_folder=args.dataset_folder, image_prep=args.test_image_prep, split="test", deg_file_path=args.deg_file_path, args=args)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # init RAM
    ram_transforms = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    RAM = ram(pretrained='./ckp/ram_swin_large_14m.pth',
            pretrained_condition=None,
            image_size=384,
            vit='swin_l')
    RAM.eval()
    RAM.to("cuda", dtype=torch.float16)

    # Prepare everything with our `accelerator`.
    TVTSR, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc = accelerator.prepare(
        TVTSR, net_disc, optimizer, optimizer_disc, dl_train, lr_scheduler, lr_scheduler_disc
    )

    net_lpips = accelerator.prepare(net_lpips)
    # renorm with image net statistics
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [TVTSR, net_disc]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # image description
                x_tgt_ram = ram_transforms(x_tgt*0.5+0.5)
                caption_r = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                with torch.no_grad():
                    positive_prompt = []
                    negative_prompt = []
                    for i in range(B):
                        ram_image = x_tgt[i,:,:,:].unsqueeze(0)
                        x_tgt_ram = ram_transforms(ram_image*0.5+0.5)
                        caption = inference(x_tgt_ram.to(dtype=torch.float16), RAM)
                        positive_prompt.append(f'{caption[0]}, {args.positive_prompt}')
                        negative_prompt.append(args.negative_prompt)

                # forward pass
                x_tgt_pred, latents_pred, prompt_embeds, neg_prompt_embeds = TVTSR(x_src, positive_prompt=positive_prompt, negative_prompt=negative_prompt, args=args)
                # Reconstruction loss
                loss_l2 = F.mse_loss(x_tgt_pred.float(), x_tgt.float(), reduction="mean") * args.lambda_l2
                loss_lpips = net_lpips(x_tgt_pred.float(), x_tgt.float()).mean() * args.lambda_lpips
                loss = loss_l2 + loss_lpips

                # latents_predict = latents_pred #vae_sd.encode(x_tgt_pred).latent_dist.sample()
                assert latents_pred.shape[2] == 128
                assert latents_pred.shape[3] == 128
                patch_size=64
                patch1 = latents_pred[:, :, :patch_size, :patch_size]    
                patch2 = latents_pred[:, :, :patch_size, patch_size:]   
                patch3 = latents_pred[:, :, patch_size:, :patch_size]   
                patch4 = latents_pred[:, :, patch_size:, patch_size:]   
                latents_predict = torch.cat([patch1, patch2, patch3, patch4], dim=0)
                prompt_embeds = torch.cat((prompt_embeds,prompt_embeds,prompt_embeds,prompt_embeds),dim=0)
                neg_prompt_embeds = torch.cat((neg_prompt_embeds,neg_prompt_embeds,neg_prompt_embeds,neg_prompt_embeds),dim=0)
                # latents_predict = latents_predict * vae_sd.config.scaling_factor
                if torch.cuda.device_count() > 1:
                    loss_kl = net_disc.module.distribution_matching_loss(net_disc.module.unet_fix, net_disc.module.unet_update, net_disc.module.sched, latents_predict, prompt_embeds, neg_prompt_embeds, args, ) * args.lambda_vsd
                else:
                    loss_kl = net_disc.distribution_matching_loss(net_disc.unet_fix, net_disc.unet_update, net_disc.sched, latents_predict, prompt_embeds, neg_prompt_embeds, args, ) * args.lambda_vsd
                loss = loss + loss_kl

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

                """
                Disc loss
                """
                if torch.cuda.device_count() > 1:
                    loss_d = net_disc.module.compute_lora_loss(latents_predict, prompt_embeds, args)*args.lambda_vsd_lora
                else:
                    loss_d = net_disc.compute_lora_loss(latents_predict, prompt_embeds, args)*args.lambda_vsd_lora
                accelerator.backward(loss_d)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(net_disc.parameters(), args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_d"] = loss_d.detach().item()
                    logs["loss_kl"] = loss_kl.detach().item()
                    logs["loss_l2"] = loss_l2.detach().item()
                    logs["loss_lpips"] = loss_lpips.detach().item()
                    progress_bar.set_postfix(**logs)

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(TVTSR).save_model(outf)

                    # compute validation set FID, L2, LPIPS, CLIP-SIM
                    if global_step % args.eval_freq == 1:
                        l_l2 = []
                        os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            x_basename = batch_val["base_name"][0]
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                # image description                                
                                positive_prompt = []
                                negative_prompt = []
                                for i in range(B):
                                    ram_image = x_src[i,:,:,:].unsqueeze(0)
                                    x_src_ram = ram_transforms(ram_image*0.5+0.5)
                                    caption = inference(x_src_ram.to(dtype=torch.float16), RAM)
                                    positive_prompt.append(f'{caption[0]}, {args.positive_prompt}')
                                    negative_prompt.append(args.negative_prompt)
                                
                                x_tgt_pred, latents_pred, _, _ = accelerator.unwrap_model(TVTSR)(x_src, positive_prompt=positive_prompt, negative_prompt=negative_prompt, args=args)
                                # save the output
                                output_pil = transforms.ToPILImage()(x_tgt_pred[0].cpu() * 0.5 + 0.5)
                                input_image = transforms.ToPILImage()(x_src[0].cpu() * 0.5 + 0.5)
                                if args.align_method == 'adain':
                                    output_pil = adain_color_fix(target=output_pil, source=input_image)
                                elif args.align_method == 'wavelet':
                                    output_pil = wavelet_color_fix(target=output_pil, source=input_image)
                                else:
                                    pass
                                outf = os.path.join(args.output_dir, "eval", f"fid_{global_step}", f"{x_basename}")
                                output_pil.save(outf)
                        
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_realsr_training()
    main(args)

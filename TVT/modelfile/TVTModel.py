import os
import requests
import sys
import copy
import random
import time
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from peft import LoraConfig
from diffusers.models.attention import FeedForward
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
p = "TVT/"
sys.path.append(p)
from VAED4.autoencoder_kl import AutoencoderKL as DiffusersVAED4

def find_filepath(directory, filename):
    matches = glob.glob(f"{directory}/**/{filename}", recursive=True)
    return matches[0] if matches else None

import yaml
def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def make_1step_sched(pretrained_model_path):
    noise_scheduler_1step = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step

def initialize_unet_TVT(rank, return_lora_module_names=False, pretrained_model_name_or_path=None, args=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    if args.use_lr_concat_lr_999noise:
        new_conv_in = torch.nn.Conv2d(8, 320, 3, 1, 1)
        new_conv_in.weight.data[:, :4, ...] = unet.conv_in.weight.data
        new_conv_in.weight.data[:, -4:, ...] = unet.conv_in.weight.data
        new_conv_in.bias.data = unet.conv_in.bias.data
        unet.conv_in = new_conv_in
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

def initialize_unet(rank, return_lora_module_names=False, pretrained_model_name_or_path=None):
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    unet.requires_grad_(False)
    unet.train()

    l_target_modules_encoder, l_target_modules_decoder, l_modules_others = [], [], []
    l_grep = ["to_k", "to_q", "to_v", "to_out.0", "conv", "conv1", "conv2", "conv_in", "conv_shortcut", "conv_out", "proj_out", "proj_in", "ff.net.2", "ff.net.0.proj"]
    for n, p in unet.named_parameters():
        if "bias" in n or "norm" in n: continue
        for pattern in l_grep:
            if pattern in n and ("down_blocks" in n or "conv_in" in n):
                l_target_modules_encoder.append(n.replace(".weight",""))
                break
            elif pattern in n and "up_blocks" in n:
                l_target_modules_decoder.append(n.replace(".weight",""))
                break
            elif pattern in n:
                l_modules_others.append(n.replace(".weight",""))
                break
    lora_conf_encoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_encoder)
    lora_conf_decoder = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_target_modules_decoder)
    lora_conf_others = LoraConfig(r=rank, init_lora_weights="gaussian",target_modules=l_modules_others)
    unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
    unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
    unet.add_adapter(lora_conf_others, adapter_name="default_others")
    if return_lora_module_names:
        return unet, l_target_modules_encoder, l_target_modules_decoder, l_modules_others
    else:
        return unet

class VSD(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path_vsd, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path_vsd, subfolder="text_encoder")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path_vsd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path_vsd, subfolder="vae")
        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path_vsd, subfolder="unet")
        self.unet_update, self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others =\
                initialize_unet(rank=args.lora_rank_unet_vsd, pretrained_model_name_or_path=args.pretrained_model_name_or_path, return_lora_module_names=True)
        self.lora_rank_unet = args.lora_rank_unet_vsd

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
                self.unet_update.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        if args.gradient_checkpointing:
            self.unet_fix.enable_gradient_checkpointing()
            self.unet_update.enable_gradient_checkpointing()

        self.text_encoder.to(accelerator.device, dtype=weight_dtype)
        self.unet_fix.to(accelerator.device, dtype=weight_dtype)
        self.unet_update.to(accelerator.device)
        self.vae.to(accelerator.device)
        
        self.text_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.unet_fix.requires_grad_(False)

    def set_eval(self):
        self.unet_fix.eval()
        self.unet.eval()
        self.unet_update.eval()

    def set_train(self):
        self.unet_update.train()
        for n, _p in self.unet_update.named_parameters():
            if "lora" in n:
                _p.requires_grad = True

    def forward(self, c_t, prompt=None, neg_prompt_tokens=None, prompt_tokens=None, deterministic=True, r=1.0, noise_map=None, args=None):

        caption_enc = self.text_encoder(prompt_tokens)[0]
        neg_caption_enc = self.text_encoder(neg_prompt_tokens)[0]

        encoded_control = self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample

        output_image = (self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample).clamp(-1, 1)

        return output_image, caption_enc, neg_caption_enc

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def compute_lora_loss(self, latents_pred, prompt_embeds, args):

        latents_pred = latents_pred.detach()
        prompt_embeds = prompt_embeds.detach()
        noise = torch.randn_like(latents_pred)
        bsz = latents_pred.shape[0]
        timesteps = torch.randint(0, self.sched.config.num_train_timesteps, (bsz,), device=latents_pred.device)
        timesteps = timesteps.long()
        noisy_latents = self.sched.add_noise(latents_pred, noise, timesteps)
        disc_pred = self.forward_latent(
            self.unet_update,
            timestep=timesteps,
            latents=noisy_latents,
            prompt_embeds=prompt_embeds
        )
        if args.snr_gamma_vsd is None:
            loss_d = F.mse_loss(disc_pred.float(), noise.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.sched, timesteps)
            if self.sched.config.prediction_type == "v_prediction":
                # Velocity objective requires that we add one to SNR values before we divide by them.
                snr = snr + 1
            mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss_d = loss.mean()

        return loss_d

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def distribution_matching_loss(
        self,
        real_model,
        fake_model,
        noise_scheduler,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(noise_scheduler.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(noise_scheduler.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noise_pred = self.forward_latent(
                fake_model,
                latents=noisy_latents,
                timestep=timestep,
                prompt_embeds=prompt_embeds.float(),
            )
            pred_fake_latents = self.eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

            noise_pred = self.forward_latent(
                real_model,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_vsd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(noise_scheduler, noise_pred, noisy_latents, timestep)

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))
        return loss

    def stopgrad(self, x):
        return x.detach()

    def save_model(self, outf):
        sd = {}
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k}
        torch.save(sd, outf)

class TVT(torch.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder").cuda()
        self.sched = make_1step_sched(args.pretrained_model_name_or_path)
        self.args = args
        self.device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        vae_ckp = torch.load(args.vae4d_path)
        vae = DiffusersVAED4.from_config(DiffusersVAED4.load_config('TVT/VAED4/config2.json'),torch_dtype=torch.float32)
        ckp_dict = {}
        for n in vae_ckp['state_dict'].keys():
            if "vaeDe4" in n:
                ckp_dict[n.replace('vaeDe4.','')]=vae_ckp['state_dict'][n]
        vae.load_state_dict(ckp_dict,strict=True)

        unet = UNet2DConditionModel.from_pretrained(args.pretrained_unet_path, subfolder="unet")
        
        if args.pretrained_path is None:
            print('==================================> randomly initiate the weight')
            unet, lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others =\
                 initialize_unet_TVT(rank=args.lora_rank_unet, pretrained_model_name_or_path=args.pretrained_unet_path, return_lora_module_names=True, args=args)
            self.lora_rank_unet = args.lora_rank_unet
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others = \
                lora_unet_modules_encoder, lora_unet_modules_decoder, lora_unet_others
        
        self.unet = unet

        if args.pretrained_path:
            print('==================================> loading pre-trained weight')
            sd = torch.load(args.pretrained_path)
            self.load_ckpt_from_state_dict(sd)

        self.unet = self.unet.cuda()
        self.vae = vae.cuda()
        self.timesteps = torch.tensor([args.time_step], device="cuda").long()
        self.text_encoder.requires_grad_(False)

    def set_eval(self):
        self.unet.eval()
        self.vae.eval()
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)

    def set_train(self):
        self.unet.train()
        for n, _p in self.unet.named_parameters():
            if "lora" in n or 'down_blocks.0' in n or 'up_blocks.3.upsamplers.0' in n or 'up_blocks.4' in n or 'conv_in' in n or 'conv_out' in n:
                _p.requires_grad = True
        for n,_p in self.vae.named_parameters():
            _p.requires_grad = False

    def encode_prompt(self, prompt):
        with torch.no_grad():
            text_input_ids = self.tokenizer(
                prompt, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            prompt_embeds = self.text_encoder(
                text_input_ids.to(self.text_encoder.device),
            )[0]
        return prompt_embeds

    def forward(self, c_t, positive_prompt=None, negative_prompt=None, args=None):
        caption_enc = self.encode_prompt(positive_prompt)
        neg_caption_enc = self.encode_prompt(negative_prompt)
        with torch.cuda.amp.autocast(enabled=False):
            encoded_control =  self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32)).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        with torch.cuda.amp.autocast(enabled=False):
            output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        return output_image.clamp(-1, 1), x_denoised, caption_enc, neg_caption_enc

    def test_forward(self, c_t, positive_prompt=None, negative_prompt=None, args=None):
        caption_enc = self.encode_prompt(positive_prompt)
        with torch.cuda.amp.autocast(enabled=False):
            encoded_control =  self.vae.encode(c_t).latent_dist.sample() * self.vae.config.scaling_factor
        
        model_pred = self.unet(encoded_control, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32)).sample
        x_denoised = self.sched.step(model_pred, self.timesteps, encoded_control, return_dict=True).prev_sample
        with torch.cuda.amp.autocast(enabled=False):
            output_image = self.vae.decode(x_denoised / self.vae.config.scaling_factor).sample
        return output_image.clamp(-1, 1)

    def save_model(self, outf):
        sd = {} 
        sd["unet_lora_encoder_modules"], sd["unet_lora_decoder_modules"], sd["unet_lora_others_modules"] =\
            self.lora_unet_modules_encoder, self.lora_unet_modules_decoder, self.lora_unet_others
        sd["rank_unet"] = self.lora_rank_unet
        sd["state_dict_unet"] = {k: v for k, v in self.unet.state_dict().items() if "lora" in k or 'down_blocks.0' in k or 'up_blocks.3.upsamplers.0' in k or 'up_blocks.4' in k or 'conv_in' in k or 'conv_out' in k}
        torch.save(sd, outf)
    
    def load_ckpt_from_state_dict(self, sd):
        # load unet lora
        lora_conf_encoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_encoder_modules"])
        lora_conf_decoder = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_decoder_modules"])
        lora_conf_others = LoraConfig(r=sd["rank_unet"], init_lora_weights="gaussian", target_modules=sd["unet_lora_others_modules"])
        self.unet.add_adapter(lora_conf_encoder, adapter_name="default_encoder")
        self.unet.add_adapter(lora_conf_decoder, adapter_name="default_decoder")
        self.unet.add_adapter(lora_conf_others, adapter_name="default_others")
        for n, p in self.unet.named_parameters():
            if "lora" in n or 'down_blocks.0' in n or 'up_blocks.3.upsamplers.0' in n or 'up_blocks.4' in n or 'conv_in' in n or 'conv_out' in n:
                p.data.copy_(sd["state_dict_unet"][n])

    def tile_forward(self, c_t, positive_prompt):

        lq = c_t
        caption_enc = self.encode_prompt(positive_prompt)
        with torch.cuda.amp.autocast(enabled=False):
            lq_latent = self.vae.encode(lq).latent_dist.sample() * self.vae.config.scaling_factor
        _, _, h, w = lq_latent.size()
        tile_size, tile_overlap = (self.args.tiled_size, self.args.tiled_overlap)

        if h * w <= self.args.tiled_size * self.args.tiled_size:
            print(f'Do not tile, cos {h * w} smaller than 96x96')
            model_pred = self.unet(lq_latent, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
        else:
            print(f'Need tile, cos {h * w} larger than 96x96')
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)
            tile_size = min(tile_size, min(h, w))
            tile_weights = self._gaussian_weights(tile_size, tile_size, 1)

            grid_rows = 0
            cur_x = 0
            while cur_x < lq_latent.size(-1):
                cur_x = max(grid_rows * tile_size-tile_overlap * grid_rows, 0)+tile_size
                grid_rows += 1

            grid_cols = 0
            cur_y = 0
            while cur_y < lq_latent.size(-2):
                cur_y = max(grid_cols * tile_size-tile_overlap * grid_cols, 0)+tile_size
                grid_cols += 1

            input_list = []
            noise_preds = []
            for row in range(grid_rows):
                noise_preds_row = []
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    # input tile dimensions
                    input_tile = lq_latent[:, :, input_start_y:input_end_y, input_start_x:input_end_x]
                    input_list.append(input_tile)

                    if len(input_list) == 1 or col == grid_cols-1:
                        input_list_t = torch.cat(input_list, dim=0)
                        # predict the noise residual
                        model_out = self.unet(input_list_t, self.timesteps, encoder_hidden_states=caption_enc.to(torch.float32),).sample
                        input_list = []
                    noise_preds.append(model_out)

            # Stitch noise predictions for all tiles
            noise_pred = torch.zeros(lq_latent.shape, device=lq_latent.device)
            contributors = torch.zeros(lq_latent.shape, device=lq_latent.device)
            # Add each tile contribution to overall latents
            for row in range(grid_rows):
                for col in range(grid_cols):
                    if col < grid_cols-1 or row < grid_rows-1:
                        # extract tile from input image
                        ofs_x = max(row * tile_size-tile_overlap * row, 0)
                        ofs_y = max(col * tile_size-tile_overlap * col, 0)
                        # input tile area on total image
                    if row == grid_rows-1:
                        ofs_x = w - tile_size
                    if col == grid_cols-1:
                        ofs_y = h - tile_size

                    input_start_x = ofs_x
                    input_end_x = ofs_x + tile_size
                    input_start_y = ofs_y
                    input_end_y = ofs_y + tile_size

                    noise_pred[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += noise_preds[row*grid_cols + col] * tile_weights
                    contributors[:, :, input_start_y:input_end_y, input_start_x:input_end_x] += tile_weights
            # Average overlapping areas with more than 1 contributor
            noise_pred /= contributors
            model_pred = noise_pred

        x_denoised = self.sched.step(model_pred, self.timesteps, lq_latent, return_dict=True).prev_sample
        with torch.cuda.amp.autocast(enabled=False):
            output_image = self.vae.decode(x_denoised.to(torch.float32) / self.vae.config.scaling_factor).sample
        
        return output_image.clamp(-1, 1)

    def _gaussian_weights(self, tile_width, tile_height, nbatches):
        """Generates a gaussian mask of weights for tile contributions"""
        from numpy import pi, exp, sqrt
        import numpy as np

        latent_width = tile_width
        latent_height = tile_height

        var = 0.01
        midpoint = (latent_width - 1) / 2  # -1 because index goes from 0 to latent_width - 1
        x_probs = [exp(-(x-midpoint)*(x-midpoint)/(latent_width*latent_width)/(2*var)) / sqrt(2*pi*var) for x in range(latent_width)]
        midpoint = latent_height / 2
        y_probs = [exp(-(y-midpoint)*(y-midpoint)/(latent_height*latent_height)/(2*var)) / sqrt(2*pi*var) for y in range(latent_height)]

        weights = np.outer(y_probs, x_probs)
        return torch.tile(torch.tensor(weights, device=self.device), (nbatches, self.unet.config.in_channels, 1, 1))
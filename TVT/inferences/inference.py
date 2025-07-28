import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
import sys
sys.path.append("TVT")
from modelfile.TVTModel import TVT
sys.path.append('TVT')
from my_utils.wavelet_color_fix import adain_color_fix, wavelet_color_fix
import glob
import open_clip
sys.path.append('./')
from ram.models.ram_lora import ram
from ram import inference_ram as inference
tensor_transforms = transforms.Compose([
                transforms.ToTensor(),
            ])

ram_transforms = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def get_validation_prompt(args, image, model, device='cuda'):
    validation_prompt = ""
    lq = tensor_transforms(image).unsqueeze(0).to(device)
    lq = ram_transforms(lq)
    captions = inference(lq, model)
    validation_prompt = f"{captions[0]}, {args.prompt},"
    
    return validation_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, default="", help='path to the input image')
    parser.add_argument('--model_name', type=str, default='realsr', help='name of the pretrained model to be used')
    parser.add_argument('--pretrained_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='', help='the directory to save the output')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--vae4d_path', type=str, default='')
    parser.add_argument('--pretrained_unet_path', type=str, default='')
    parser.add_argument("--process_size", type=int, default=512)
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--align_method", type=str, choices=['wavelet', 'adain', 'nofix'], default='adain')
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="")
    parser.add_argument('--ram_ft_path', type=str, default=None) # 
    parser.add_argument('--prompt', type=str, default='', help='positive prompts')
    parser.add_argument('--negprompt', type=str, default='', help='negative prompts')
    parser.add_argument('--unet_fuse_lora', type=bool, default=False)
    parser.add_argument("--cfg", type=float, default=0)
    parser.add_argument('--time_step', type=int, default=999)
    parser.add_argument('--tiled_size', type=int, default=192)
    parser.add_argument('--tiled_overlap', type=int, default=64)

    args = parser.parse_args()

    # initialize the model
    TVTSR = TVT(args)
    TVTSR.set_eval()
    TVTSR.unet.set_adapter(['default_encoder', 'default_decoder', 'default_others'])

    if os.path.isdir(args.input_image):
        image_names = sorted(glob.glob(f'{args.input_image}/*.png'))
    else:
        image_names = [args.input_image]

    print("=== use ram ===")
    model_vlm = ram(pretrained='./ckp/ram_swin_large_14m.pth',
            pretrained_condition=args.ram_ft_path,
            image_size=384,
            vit='swin_l')
    model_vlm.eval()
    model_vlm.to("cuda")
        
    # make the output dir
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'There are {len(image_names)} images.')
    for image_name in image_names:
        
        # make sure that the input image is a multiple of 8
        input_image = Image.open(image_name).convert('RGB')
        ori_width, ori_height = input_image.size
        rscale = args.upscale
        resize_flag = False
        if ori_width < args.process_size//rscale or ori_height < args.process_size//rscale:
            scale = (args.process_size//rscale)/min(ori_width, ori_height)
            input_image = input_image.resize((int(scale*ori_width), int(scale*ori_height)))
            resize_flag = True
        input_image = input_image.resize((input_image.size[0]*rscale, input_image.size[1]*rscale))

        new_width = input_image.width - input_image.width % 8
        new_height = input_image.height - input_image.height % 8
        input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
        bname = os.path.basename(image_name)
        
        # get caption
        validation_prompt = get_validation_prompt(args, input_image, model_vlm)
        print(image_name, validation_prompt)

        # translate the image
        with torch.no_grad():
            c_t = F.to_tensor(input_image).unsqueeze(0).cuda()*2-1
            output_image = TVTSR.test_forward(c_t, positive_prompt=[validation_prompt])
            output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)
            if args.align_method == 'adain':
                output_pil = adain_color_fix(target=output_pil, source=input_image)
            elif args.align_method == 'wavelet':
                output_pil = wavelet_color_fix(target=output_pil, source=input_image)
            else:
                pass
            if resize_flag:
                output_pil.resize((int(args.upscale*ori_width), int(args.upscale*ori_height)))

        output_pil.save(os.path.join(args.output_dir, bname))
    
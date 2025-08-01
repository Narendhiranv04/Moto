import os
import sys

# Manually add the repository root to the Python path so that local
# packages can be imported without relying on `pyrootutils`.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

import argparse
import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import argparse
import os
import json
import math
from glob import glob
from functools import partial

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import numpy as np

from common.models.model_utils import load_model


def get_image_processor(vision_processor_config):
    input_size = (vision_processor_config['size'], vision_processor_config['size'])
    rgb_mean = vision_processor_config['image_mean']
    rgb_std = vision_processor_config['image_std']
    return T.Compose([
        T.ToTensor(),
        T.Resize(input_size, interpolation=Image.BICUBIC),
        T.Normalize(rgb_mean, rgb_std)
    ])


def get_image_seq_post_processor(image_seq, image_std, image_mean):
    image_mean = torch.tensor(image_mean).view(1, 3, 1, 1)
    image_std = torch.tensor(image_std).view(1, 3, 1, 1)
    image_seq = image_seq * image_std + image_mean
    image_seq = torch.clamp(image_seq, min=0, max=1)
    return list(map(T.ToPILImage(), image_seq.unbind(dim=0)))


def save_compare_image(gt_img, pred_img, save_path):
    w, h = gt_img.size
    canvas = Image.new('RGB', (w * 2, h))
    canvas.paste(gt_img, (0, 0))
    canvas.paste(pred_img, (w, 0))
    canvas.save(save_path)


def save_video(frames, save_path, post_process, fps=4):
    images = post_process(frames.cpu())
    if not images:
        return
    w, h = images[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
    for img in images:
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


def save_compare_video(gt_frames, pred_frames, save_path, post_process, fps=4):
    gt_imgs = post_process(gt_frames.cpu())
    pred_imgs = post_process(pred_frames.cpu())
    if not gt_imgs or not pred_imgs:
        return
    w, h = gt_imgs[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_path, fourcc, fps, (w * 2, h))
    for g, p in zip(gt_imgs, pred_imgs):
        canvas = Image.new('RGB', (w * 2, h))
        canvas.paste(g, (0, 0))
        canvas.paste(p, (w, 0))
        frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()

def save_overview_image(gt_frames, pred_frames, gt_ids_list, pred_ids_list,
                        save_path, post_process):
    gt_imgs = post_process(gt_frames.cpu())
    pred_imgs = post_process(pred_frames.cpu())
    if not gt_imgs or not pred_imgs:
        return

    from PIL import ImageDraw, ImageFont

    w, h = gt_imgs[0].size
    font = ImageFont.load_default()
    try:
        text_h = font.getsize("0")[1] + 2
    except AttributeError:
        # Pillow >=10 removed getsize; use getbbox instead
        bbox = font.getbbox("0")
        text_h = (bbox[3] - bbox[1]) + 2
    row_h = h + text_h
    num_steps = len(gt_imgs)

    canvas = Image.new('RGB', (w * num_steps, row_h * 2), 'white')
    draw = ImageDraw.Draw(canvas)

    for idx, (img, ids) in enumerate(zip(gt_imgs, gt_ids_list)):
        canvas.paste(img, (idx * w, 0))
        draw.text((idx * w, h), " ".join(map(str, ids)), fill='black', font=font)

    for idx, (img, ids) in enumerate(zip(pred_imgs, pred_ids_list)):
        canvas.paste(img, (idx * w, row_h))
        draw.text((idx * w, row_h + h), " ".join(map(str, ids)), fill='black', font=font)

    canvas.save(save_path)


def evaluate_video(video_path, lang_goal, moto_gpt, latent_motion_tokenizer,
                   lang_tokenizer, image_processor, image_seq_post_processor,
                   seq_len, delta_t, step_interval, output_dir):
    device = moto_gpt.device
    video = cv2.VideoCapture(video_path)
    video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(0, video_len, delta_t):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = image_processor(Image.fromarray(frame).convert('RGB'))
        frames.append(frame)
    video.release()

    frames = torch.stack(frames).to(device)
    initial_frame = frames[0]
    subsequent_frames = frames[1:]
    exact_num_gen_frames = subsequent_frames.shape[0]

    gt_latent_motion_ids = latent_motion_tokenizer(
        cond_pixel_values=frames[:-1],
        target_pixel_values=frames[1:],
        return_motion_token_ids_only=True
    )

    lang_inputs = lang_tokenizer(lang_goal, return_tensors='pt', padding=True)
    tokenized_text = lang_inputs.input_ids.to(device)
    lang_attention_mask = lang_inputs.attention_mask.to(device)

    attention_mask = torch.ones(1, seq_len).long().to(device)
    latent_mask = attention_mask
    dummy_latent_motion_ids = torch.zeros(
        (1, seq_len, gt_latent_motion_ids.shape[-1]), dtype=torch.long, device=device)

    decoding_kwargs = {
        'temperature': 1.0,
        'sample': False,
        'top_k': 0,
        'top_p': 1.0,
        'beam_size': 1,
        'parallel': False,
    }

    gen_iter_num = math.ceil(exact_num_gen_frames / seq_len)
    cur_cond_pixel_values = initial_frame.unsqueeze(0)
    cur_latent_motion_ids = dummy_latent_motion_ids.clone()
    cur_initial_frame = initial_frame.unsqueeze(0).unsqueeze(0)

    frame_preds = []
    latent_motion_id_preds = []
    for _ in range(gen_iter_num):
        for buffer_len in range(1, seq_len + 1):
            pred = moto_gpt(
                rgb=cur_initial_frame,
                language=tokenized_text,
                attention_mask=attention_mask,
                latent_motion_ids=cur_latent_motion_ids,
                latent_mask=latent_mask,
                train=False,
                lang_attention_mask=lang_attention_mask,
                buffer_len=buffer_len,
                **decoding_kwargs,
            )
            cur_ids = pred['latent_motion_id_preds']
            cur_latent_motion_ids[:, buffer_len - 1] = cur_ids
            cur_frame = latent_motion_tokenizer.decode_image(
                cond_pixel_values=cur_cond_pixel_values,
                given_motion_token_ids=cur_ids.reshape(-1, cur_ids.shape[-1])
            )["recons_pixel_values"]
            cur_cond_pixel_values = cur_frame
            frame_preds.append(cur_frame.cpu())
            latent_motion_id_preds.append(cur_ids.cpu())
        cur_initial_frame = cur_frame.unsqueeze(1)

    frame_preds = torch.cat(frame_preds, dim=0)[:exact_num_gen_frames]
    latent_motion_id_preds = torch.cat(latent_motion_id_preds, dim=0)[:exact_num_gen_frames]

    gt_ids_all = gt_latent_motion_ids.detach().cpu()
    pred_ids_all = latent_motion_id_preds.detach().cpu()


    print(f"==> {os.path.basename(video_path)} : {lang_goal}")
    for i in range(0, exact_num_gen_frames, step_interval):
        gt_ids = gt_latent_motion_ids[i].detach().cpu().tolist()
        pred_ids = latent_motion_id_preds[i].detach().cpu().tolist()

        gt_vec = latent_motion_tokenizer.vector_quantizer.get_codebook_entry(
            gt_latent_motion_ids[i:i+1].long().to(device)
        ).view(-1)
        pred_vec = latent_motion_tokenizer.vector_quantizer.get_codebook_entry(
            latent_motion_id_preds[i:i+1].long().to(device)
        ).view(-1)
        rmse = torch.sqrt(torch.mean((pred_vec - gt_vec) ** 2)).item()
        print(
            (
                f"Step {i + 1}/{exact_num_gen_frames} | "
                f"GT IDs: {gt_ids} | Pred IDs: {pred_ids} | RMSE: {rmse:.6f}"
            )
        )

        gt_img = image_seq_post_processor(subsequent_frames[i : i + 1].cpu())[0]
        pred_img = image_seq_post_processor(frame_preds[i : i + 1])[0]

        save_compare_image(
            gt_img,
            pred_img,
            os.path.join(
                output_dir,
                f"{os.path.basename(video_path).split('.')[0]}_step_{i + 1}.png",

            ),
        )

    # save full videos
    pred_full = torch.cat([
        initial_frame.cpu().unsqueeze(0),
        frame_preds
    ], dim=0)
    gt_full = frames.cpu()

    base_name = os.path.basename(video_path).split(".")[0]
    save_video(
        pred_full,
        os.path.join(output_dir, f"{base_name}_pred.mp4"),
        image_seq_post_processor,
    )
    save_video(
        gt_full,
        os.path.join(output_dir, f"{base_name}_gt.mp4"),
        image_seq_post_processor,
    )
    save_compare_video(
        gt_full,
        pred_full,
        os.path.join(output_dir, f"{base_name}_compare.mp4"),
        image_seq_post_processor,
    )

    save_overview_image(
        subsequent_frames,
        frame_preds,
        gt_ids_all.tolist(),
        pred_ids_all.tolist(),
        os.path.join(output_dir, f"{base_name}_overview.png"),
        image_seq_post_processor,
    )


def main(args):
    print(f"loading Moto-GPT from {args.moto_gpt_path} ...")
    moto_gpt = load_model(args.moto_gpt_path)
    moto_gpt_config = moto_gpt.config
    moto_gpt = moto_gpt.cuda()
    moto_gpt.eval()

    lang_tokenizer = AutoTokenizer.from_pretrained(
        moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    vision_cfg = json.load(open(get_file_from_repo(
        moto_gpt_config['model_vision']['pretrained_model_name_or_path'], FEATURE_EXTRACTOR_NAME)))
    image_processor = get_image_processor(vision_cfg)
    image_seq_post_processor = partial(
        get_image_seq_post_processor,
        image_std=vision_cfg['image_std'],
        image_mean=vision_cfg['image_mean']
    )

    print(f"loading Latent Motion Tokenizer from {args.latent_motion_tokenizer_path} ...")
    latent_motion_tokenizer = load_model(args.latent_motion_tokenizer_path)
    latent_motion_tokenizer = latent_motion_tokenizer.cuda()
    latent_motion_tokenizer.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.input_dir, 'lang_annotations.json')) as f:
        lang_annotations = json.load(f)

    video_dir = os.path.join(args.input_dir, 'videos')
    videos = sorted(glob(os.path.join(video_dir, '*.mp4')))
    for video_path in tqdm(videos):
        base = os.path.basename(video_path)
        lang_goal = lang_annotations.get(base, '')
        evaluate_video(
            video_path=video_path,
            lang_goal=lang_goal,
            moto_gpt=moto_gpt,
            latent_motion_tokenizer=latent_motion_tokenizer,
            lang_tokenizer=lang_tokenizer,
            image_processor=image_processor,
            image_seq_post_processor=image_seq_post_processor,
            seq_len=moto_gpt_config['sequence_length'],
            delta_t=args.delta_t,
            step_interval=args.print_every,
            output_dir=args.output_dir,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moto_gpt_path', type=str, required=True)
    parser.add_argument('--latent_motion_tokenizer_path', type=str, required=True)
    parser.add_argument('--delta_t', type=int, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--print_every', type=int, default=5)
    args = parser.parse_args()
    main(args)

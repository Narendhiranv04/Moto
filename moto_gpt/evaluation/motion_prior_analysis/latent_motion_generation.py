import pyrootutils
import os
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
from omegaconf import OmegaConf
import hydra
import torchvision.transforms as T
import json
import argparse
from tqdm import tqdm
from glob import glob
import cv2
from PIL import Image, ImageFont, ImageDraw
import torch
import math
from functools import partial
from transformers import AutoTokenizer
from transformers.utils import FEATURE_EXTRACTOR_NAME, get_file_from_repo
import numpy as np
from collections import defaultdict
from common.models.model_utils import load_model
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from io import BytesIO

def get_image_processor(vision_processor_config):
    input_size = (vision_processor_config['size'], vision_processor_config['size'])
    rgb_mean = vision_processor_config['image_mean']
    rgb_std = vision_processor_config['image_std']
    image_transform = T.Compose([
        T.ToTensor(),
        T.Resize(input_size, interpolation=Image.BICUBIC),
        T.Normalize(rgb_mean, rgb_std)])
    return image_transform

def get_image_seq_post_processor(image_seq, image_std, image_mean):
    image_mean = torch.tensor(image_mean).view(1,3,1,1)
    image_std = torch.tensor(image_std).view(1,3,1,1)
    image_seq = image_seq * image_std + image_mean # (t, c, h, w)
    image_seq = torch.clamp(image_seq, min=0, max=1)
    image_seq = list(map(T.ToPILImage(), image_seq.unbind(dim=0)))
    return image_seq


def visualization(
        lang_goal,
        orig_video, 
        decoding_mode2preds,
        path, 
        image_seq_post_processor
    ):
    n_cols, c, h, w = orig_video.shape
    n_rows = len(decoding_mode2preds)+1
    h = h + 30

    orig_video = image_seq_post_processor(orig_video)
    initial_frame = orig_video[0]
    gt_subsequent_frames = orig_video[1:]
    for decoding_mode, preds in decoding_mode2preds.items():
        preds['frame_preds'] = image_seq_post_processor(preds['frame_preds'])
        preds['latent_motion_id_preds'] = preds['latent_motion_id_preds'].numpy().tolist()

    font_path = os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSans.ttf')
    font = ImageFont.truetype(font_path, size=12)
    compare_img = Image.new('RGB', size=(n_cols*w, n_rows*h))
    draw_compare_img = ImageDraw.Draw(compare_img)
    
    for i in range(n_rows):
        compare_img.paste(initial_frame, box=(0, i*h))

    for j in range(n_cols-1):
        compare_img.paste(gt_subsequent_frames[j], box=((j+1)*w, 0))

        for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
            compare_img.paste(preds['frame_preds'][j], box=((j+1)*w, (i+1)*h))
            draw_compare_img.text(((j+1)*w, (i+2)*h-20), f"{preds['latent_motion_id_preds'][j]}", font=font, fill=(0, 255, 0))
            
            if j == 0:
                draw_compare_img.text((0, (i+2)*h-20), f"{decoding_mode}", font=font, fill=(0, 255, 0))

    draw_compare_img.text((0, h-20), f"{lang_goal}", font=font, fill=(0, 255, 0))
    compare_img.save(f"{path}-{'_'.join(lang_goal.split())}.png")


    h = h - 30
    fps = 4
    for i, (decoding_mode, preds) in enumerate(decoding_mode2preds.items()):
        output_video_path = f"{path}-{'_'.join(lang_goal.split())}-{decoding_mode}.mp4"
        images = preds['frame_preds']
        images = [initial_frame] + images

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

        for image in images:
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            video_writer.write(image_cv)
        video_writer.release()

def tsne_video(vectors, path, fps=4):
    """Save a t-SNE trajectory video for the given vectors."""
    n_samples = len(vectors)
    if n_samples < 2:
        print("TSNE skipped: not enough samples")
        return
    perplexity = min(30, n_samples - 1)
    try:
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        coords = tsne.fit_transform(vectors)
    except Exception as e:
        print(f"TSNE failed: {e}")
        return

    w, h = 400, 400
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(len(coords)):
        plt.figure(figsize=(4,4))
        plt.scatter(coords[:i+1,0], coords[:i+1,1], c=np.arange(i+1), cmap='viridis')
        plt.plot(coords[:i+1,0], coords[:i+1,1], c='gray')
        plt.axis('off')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        image = Image.open(buf)
        image = image.resize((w, h))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


def latent_speed_curve(vectors, path, delta_t):
    """Plot Euclidean distance between consecutive vectors."""
    if len(vectors) < 2:
        print("Speed curve skipped: not enough samples")
        return
    speeds = np.linalg.norm(np.diff(vectors, axis=0), axis=1)
    x = np.arange(1, len(vectors)) * delta_t
    plt.figure()
    plt.plot(x, speeds, marker='o')
    plt.xlabel('Frame index')
    plt.ylabel('Latent speed')
    plt.title('Latent-speed curve')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def feature_heatmap(vectors, path):
    """Save heatmap of latent dimensions over time."""
    if len(vectors) == 0:
        print("Heatmap skipped: no vectors")
        return
    plt.figure(figsize=(6, 4))
    plt.imshow(np.array(vectors).T, aspect='auto', cmap='viridis')
    plt.xlabel('Frame index')
    plt.ylabel('Latent dim')
    plt.title('Feature heat-map')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def principal_component_ribbon(vectors, path, delta_t):
    """Plot PC1 over time colored by PC2."""
    if len(vectors) == 0:
        print("PCA ribbon skipped: no vectors")
        return
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    x = np.arange(len(vectors)) * delta_t
    plt.figure()
    sc = plt.scatter(x, coords[:, 0], c=coords[:, 1], cmap='viridis')
    plt.plot(x, coords[:, 0], color='gray', alpha=0.5)
    plt.xlabel('Frame index')
    plt.ylabel('PC1')
    plt.title('Principal-component ribbon')
    plt.colorbar(sc, label='PC2')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def tsne_trajectory_plot(f, episode_id, save_path):
    """Plot a connected t-SNE trajectory with time-coded colours."""
    X = f.detach().cpu().numpy()
    T = X.shape[0]
    if T < 2:
        print("t-SNE trajectory skipped: not enough samples")
        return
    perplexity = min(30, max(5, T // 3))
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, metric='cosine', init='pca', random_state=0)
        Z = tsne.fit_transform(X)
    except Exception as e:
        print(f"t-SNE trajectory failed: {e}")
        return

    norm_time = np.linspace(0, 1, T - 1)
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots()
    for i in range(T - 1):
        ax.plot(Z[i:i+2, 0], Z[i:i+2, 1], color=cmap(norm_time[i]), linewidth=2, alpha=0.8)
    ax.scatter(Z[0, 0], Z[0, 1], c='white', edgecolors='black', s=80, label='start', zorder=3)
    ax.scatter(Z[-1, 0], Z[-1, 1], c='black', s=80, label='end', zorder=3)
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='normalized time')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Episode {episode_id} latent-space trajectory")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def tsne_cluster_plot(task_to_vecs, save_dir):
    """Run t-SNE on all frames and plot trajectories per task."""
    all_vecs = []
    meta = []  # (task, episode_idx, frame_idx, norm_time)
    for task, vec_list in task_to_vecs.items():
        for epi_idx, vec in enumerate(vec_list):
            v = vec.detach().cpu().numpy()
            T = v.shape[0]
            all_vecs.append(v)
            for t in range(T):
                norm_t = t / (T - 1) if T > 1 else 0.0
                meta.append((task, epi_idx, t, norm_t))

    if len(all_vecs) < 2:
        print("Cluster t-SNE skipped: not enough samples")
        return

    X = np.concatenate(all_vecs, axis=0)
    perplexity = min(30, max(5, len(X) // 3))
    try:
        Z = TSNE(n_components=2, perplexity=perplexity, metric='cosine', init='pca', random_state=0).fit_transform(X)
    except Exception as e:
        print(f"Cluster t-SNE failed: {e}")
        return

    os.makedirs(save_dir, exist_ok=True)
    cmap = plt.get_cmap('viridis')
    task_names = list(task_to_vecs.keys())
    for task in task_names:
        fig, ax = plt.subplots(figsize=(6, 6))
        task_indices = [i for i, m in enumerate(meta) if m[0] == task]
        if not task_indices:
            plt.close(fig)
            continue
        # group by episode
        ep_groups = defaultdict(list)
        for idx in task_indices:
            _, epi_idx, frame_idx, norm_t = meta[idx]
            ep_groups[epi_idx].append((frame_idx, Z[idx], norm_t))
        for ep_id in sorted(ep_groups.keys()):
            pts = sorted(ep_groups[ep_id], key=lambda x: x[0])
            coords = np.array([p[1] for p in pts])
            times = [p[2] for p in pts]
            for i in range(len(coords) - 1):
                ax.plot(coords[i:i+2, 0], coords[i:i+2, 1], color=cmap(times[i]), linewidth=2, alpha=0.8)
            ax.scatter(coords[0, 0], coords[0, 1], c='white', edgecolors='black', s=60, zorder=3)
            ax.scatter(coords[-1, 0], coords[-1, 1], c='black', s=60, zorder=3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='normalized time')
        ax.set_xlabel('t-SNE-1')
        ax.set_ylabel('t-SNE-2')
        ax.set_title(f"t-SNE trajectory for task: {task}")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{task}_tsne.png"))
        plt.close(fig)

def _subsequence_dtw(query, reference):
    """Subsequence DTW distance between query and reference.

    Returns the minimal cost, start index in reference, end index and the path.
    Both query and reference are numpy arrays of shape (T, D).
    """
    m, d = query.shape
    n = reference.shape[0]
    dist = np.linalg.norm(query[:, None, :] - reference[None, :, :], axis=2)
    dp = np.full((m + 1, n + 1), np.inf)
    dp[0, :] = 0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = dist[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])

    j_min = np.argmin(dp[m, 1:]) + 1
    cost = dp[m, j_min]
    path = []
    i, j = m, j_min
    while i > 0:
        path.append((i - 1, j - 1))
        choices = [dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1]]
        arg = np.argmin(choices)
        if arg == 0:
            i -= 1
            j -= 1
        elif arg == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    start = path[0][1]
    end = path[-1][1]
    return cost, start, end, path

def _path_cosine_similarity(query, reference, path):
    cos = []
    for i, j in path:
        a = query[i]
        b = reference[j]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            cos.append(0.0)
        else:
            cos.append(float(np.dot(a, b) / denom))
    return float(np.mean(cos)) if cos else 0.0

def aligned_cosine_similarities(task_to_vecs):
    """Compute cosine similarity between tasks using DTW alignment."""
    tasks = list(task_to_vecs.keys())
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            sims = []
            for seq_a in task_to_vecs[tasks[i]]:
                a = seq_a.detach().cpu().numpy()
                for seq_b in task_to_vecs[tasks[j]]:
                    b = seq_b.detach().cpu().numpy()
                    _, _, _, path = _subsequence_dtw(a, b)
                    sims.append(_path_cosine_similarity(a, b, path))
            if sims:
                print(f"Aligned cosine similarity between {tasks[i]} and {tasks[j]}: {np.mean(sims):.6f}")


def _visualize_snippet_match(query_seq, q_start, ref_seq, r_start, save_path):
    """Visualize two matched snippets within their full trajectories."""
    snippet_len = 3
    vectors = np.concatenate([query_seq, ref_seq], axis=0)
    if vectors.shape[0] < 2:
        return

    coords = PCA(n_components=2).fit_transform(vectors)
    q_full = coords[: len(query_seq)]
    r_full = coords[len(query_seq) :]

    q_coords = q_full[q_start : q_start + snippet_len]
    r_coords = r_full[r_start : r_start + snippet_len]

    plt.figure()
    plt.plot(
        q_full[:, 0],
        q_full[:, 1],
        linestyle=':',
        color='red',
        alpha=0.5,
        label='query full',
    )
    plt.plot(
        r_full[:, 0],
        r_full[:, 1],
        linestyle=':',
        color='blue',
        alpha=0.5,
        label='reference full',
    )
    plt.plot(q_coords[:, 0], q_coords[:, 1], '-o', color='red', label='query snippet')
    plt.plot(r_coords[:, 0], r_coords[:, 1], '-o', color='blue', label='reference snippet')
    plt.annotate('', xy=q_coords[-1], xytext=q_coords[0], arrowprops=dict(arrowstyle='->', color='red', lw=2))
    plt.annotate('', xy=r_coords[-1], xytext=r_coords[0], arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _snippet_match_video(query_frames, q_start, ref_frames, r_start, save_path, post_process):
    """Save an MP4 video of two matched snippets shown side by side.

    This purely serves as a visual aid using decoded frames. The latent
    matching logic never relies on these frames.
    """
    snippet_len = 3
    q_clip = query_frames[q_start : q_start + snippet_len]
    r_clip = ref_frames[r_start : r_start + snippet_len]

    q_imgs = post_process(q_clip)
    r_imgs = post_process(r_clip)
    if not q_imgs or not r_imgs:
        return

    w, h = q_imgs[0].size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(save_path, fourcc, 1, (w * 2, h))
    for qi, ri in zip(q_imgs, r_imgs):
        canvas = Image.new("RGB", (w * 2, h))
        canvas.paste(qi, (0, 0))
        canvas.paste(ri, (w, 0))
        frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    video_writer.release()


def _embed_snippet(snippet):
    """Embed a 3-frame snippet with mean and delta features."""
    snippet = np.asarray(snippet)
    mean_feat = snippet.mean(axis=0)
    delta_feat = snippet[-1] - snippet[0]
    return np.concatenate([mean_feat, delta_feat], axis=0)


def _build_faiss_index(embeddings):
    """Build a FAISS index if possible, otherwise fall back to brute force."""
    try:
        import faiss  # type: ignore

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception as e:  # pragma: no cover - faiss may not be installed
        print(f"FAISS unavailable ({e}); falling back to brute force index")

        embeddings = embeddings.astype("float32")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        normalized = embeddings / norms

        class BruteForceIndex:
            def __init__(self, embs):
                self.embs = embs

            def search(self, q, k):
                q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)
                sims = np.dot(self.embs, q_norm.T).T
                idx = np.argsort(-sims, axis=1)[:, :k]
                dist = np.take_along_axis(sims, idx, axis=1)
                return dist.astype("float32"), idx.astype("int64")

        return BruteForceIndex(normalized)


def subtrajectory_faiss_analysis(task_to_vecs, task_to_frames, save_dir, post_process, delta_threshold=1e-3, top_k=5):
    """Compare 3-frame latent snippets across tasks.

    Embeddings are computed solely from latent vectors. Decoded frames are
    optional and used only for the saved MP4 visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "matches.log")
    log_f = open(log_path, "w")

    embeddings = []
    meta = []
    snippets = []
    full_seqs = []
    full_frames = []
    tasks = list(task_to_vecs.keys())

    for task in tasks:
        frame_list = task_to_frames.get(task, [])
        for epi_idx, seq in enumerate(task_to_vecs[task]):
            arr = seq.detach().cpu().numpy()
            frames_arr = frame_list[epi_idx] if epi_idx < len(frame_list) else None
            for t_idx in range(len(arr) - 2):
                snippet = arr[t_idx : t_idx + 3]
                emb = _embed_snippet(snippet)
                if np.linalg.norm(emb[arr.shape[1] :]) < delta_threshold:
                    continue
                embeddings.append(emb)
                meta.append((task, epi_idx, t_idx))
                snippets.append(snippet)
                full_seqs.append(arr)
                full_frames.append(frames_arr)

    if not embeddings:
        return

    embeddings = np.stack(embeddings).astype("float32")
    index = _build_faiss_index(embeddings.copy())

    try:
        import faiss
        faiss.write_index(index, os.path.join(save_dir, "subaction.index"))
    except Exception:
        pass
    with open(os.path.join(save_dir, "subaction_meta.json"), "w") as f:
        json.dump([
            {"task": t, "episode": int(e), "step": int(s)} for t, e, s in meta
        ], f)

    for i, (task, epi, step) in enumerate(meta):
        D, I = index.search(embeddings[i : i + 1], top_k + 1)
        best_j = None
        best_sim = -np.inf
        for j in I[0]:
            if j == i:
                continue
            b_task, b_epi, b_step = meta[j]
            if b_task == task:
                continue
            _, _, _, path = _subsequence_dtw(snippets[i], snippets[j])
            sim = _path_cosine_similarity(snippets[i], snippets[j], path)
            if sim > best_sim:
                best_sim = sim
                best_j = j
        if best_j is not None:
            b_task, b_epi, b_step = meta[best_j]
            msg = (
                f"Snippet {task} ep{epi} frames {step}-{step+2} -> "
                f"{b_task} ep{b_epi} frames {b_step}-{b_step+2} similarity: {best_sim:.4f}"
            )
            print(msg)
            log_f.write(msg + "\n")

            match_dir = os.path.join(
                save_dir,
                f"{task}_ep{epi}_step{step}_to_{b_task}_ep{b_epi}_step{b_step}"
            )
            os.makedirs(match_dir, exist_ok=True)
            img_path = os.path.join(match_dir, "match.png")
            video_path = os.path.join(match_dir, "match.mp4")
            _visualize_snippet_match(
                full_seqs[i],
                step,
                full_seqs[best_j],
                b_step,
                img_path,
            )
            if full_frames[i] is not None and full_frames[best_j] is not None:
                _snippet_match_video(
                    full_frames[i],
                    step,
                    full_frames[best_j],
                    b_step,
                    video_path,
                    post_process,
                )

    log_f.close()

def inference(
        moto_gpt,
        latent_motion_tokenizer,
        lang_tokenizer,
        image_processor,
        image_seq_post_processor,
        delta_t,
        moto_gpt_seq_len,
        input_dir,
        output_dir
    ):

    device = moto_gpt.device
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(input_dir, "lang_annotations.json")) as f:
        lang_annotations = json.load(f)

    # use defaultdicts so new tasks are automatically initialized
    metrics = defaultdict(lambda: defaultdict(list))

    video_dir = os.path.join(input_dir, "videos")
    for video_path in tqdm(glob(os.path.join(video_dir, "*.mp4"))):
        video_basename = os.path.basename(video_path)
        video = cv2.VideoCapture(video_path)
        video_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(0, video_len, delta_t):
            video.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = image_processor(Image.fromarray(frame).convert("RGB"))
            frames.append(frame)

        video.release()
        frames = torch.stack(frames).to(device) # (f, c, h, w)
        initial_frame = frames[0]
        subsequent_frames = frames[1:]
        exact_num_gen_frames = subsequent_frames.shape[0]

        gt_latent_motion_ids = latent_motion_tokenizer(
            cond_pixel_values=frames[:-1],
            target_pixel_values=frames[1:],
            return_motion_token_ids_only=True
        )

        recons_subsequent_frames = latent_motion_tokenizer.decode_image(
            cond_pixel_values=frames[:-1],
            given_motion_token_ids=gt_latent_motion_ids
        )["recons_pixel_values"] # (exact_num_gen_frames, c, h, w)


        decoding_mode2preds = {
            "ground_truth_recons": {
                "frame_preds": recons_subsequent_frames.detach().cpu(),
                "latent_motion_id_preds": gt_latent_motion_ids.detach().cpu()
            }
        }

        decoding_mode2latent_motion_decoding_kwargs = {
            "sampleFalse_beam1": {
                "temperature": 1.0,
                "sample": False,
                "top_k": 0,
                "top_p": 1.0,
                "beam_size": 1, 
                "parallel": False
            },

            "sampleTrue_beam5": {
                "temperature": 1.0, 
                "sample": True, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },

            "sampleFalse_beam5": {
                "temperature": 1.0, 
                "sample": False, 
                "top_k": 0, 
                "top_p": 1.0,
                "beam_size": 5, 
                "parallel": False
            },
        }

        
        lang_goal = lang_annotations[video_basename]
        lang_inputs = lang_tokenizer(lang_goal, return_tensors='pt', padding=True)
        tokenized_text = lang_inputs.input_ids.to(device)
        lang_attention_mask = lang_inputs.attention_mask.to(device)

        attention_mask = torch.ones(1, moto_gpt_seq_len).long().to(device)
        dummy_latent_motion_ids = torch.zeros((1, moto_gpt_seq_len, gt_latent_motion_ids.shape[-1])).long().to(device)
        latent_mask = attention_mask

        for decoding_mode, latent_motion_decoding_kwargs in decoding_mode2latent_motion_decoding_kwargs.items():
            gen_iter_num = math.ceil(exact_num_gen_frames / moto_gpt_seq_len)

            frame_preds = []
            latent_motion_id_preds = []

            cur_cond_pixel_values = initial_frame.unsqueeze(0) # (b, c, h, w)
            cur_latent_motion_ids = dummy_latent_motion_ids.clone() # (b, moto_gpt_seq_len, per_latent_motion_len)

            cur_initial_frame = initial_frame.unsqueeze(0).unsqueeze(0) # (b, 1, c, h, w)

            
            for _ in range(gen_iter_num):
                for buffer_len in range(1, moto_gpt_seq_len+1):
                    pred = moto_gpt(
                        rgb=cur_initial_frame, 
                        language=tokenized_text,
                        attention_mask=attention_mask,
                        latent_motion_ids=cur_latent_motion_ids,
                        latent_mask=latent_mask,
                        train=False,
                        lang_attention_mask=lang_attention_mask,
                        buffer_len=buffer_len,
                        **latent_motion_decoding_kwargs,
                    )
                    cur_latent_motion_id_preds = pred['latent_motion_id_preds'] # (b, per_latent_motion_len)
                    cur_latent_motion_ids[:,buffer_len-1] = cur_latent_motion_id_preds
                    cur_frame_preds = latent_motion_tokenizer.decode_image(
                        cond_pixel_values=cur_cond_pixel_values,
                        given_motion_token_ids=cur_latent_motion_id_preds.reshape(-1, cur_latent_motion_id_preds.shape[-1]) # (b, per_latent_motion_len)
                    )["recons_pixel_values"] # (b, c, h, w)
                    cur_cond_pixel_values = cur_frame_preds
                    frame_preds.append(cur_frame_preds.detach().cpu())
                    latent_motion_id_preds.append(cur_latent_motion_id_preds.detach().cpu())

                cur_initial_frame = cur_frame_preds.unsqueeze(1) # (b, 1, c, h, w)

            frame_preds = torch.cat(frame_preds, dim=0)[:exact_num_gen_frames]
            latent_motion_id_preds = torch.cat(latent_motion_id_preds, dim=0)[:exact_num_gen_frames]
            decoding_mode2preds[decoding_mode] = {
                "frame_preds": frame_preds.detach().cpu(),
                "latent_motion_id_preds": latent_motion_id_preds.detach().cpu()
            }

            if decoding_mode == "sampleFalse_beam1":
                gt_vec = latent_motion_tokenizer.vector_quantizer.get_codebook_entry(gt_latent_motion_ids.long().to(device))
                gt_vec = gt_vec.mean(dim=1)
                pred_vec = latent_motion_tokenizer.vector_quantizer.get_codebook_entry(
                    latent_motion_id_preds.long().to(device)
                )
                pred_vec = pred_vec.mean(dim=1)
                rmse = torch.sqrt(((pred_vec - gt_vec) ** 2).mean(dim=1))
                print(f"Task: {lang_goal}")
                print("RMSE per step:", rmse)
                metrics["task_to_rmses"][lang_goal].append(rmse.cpu())
                metrics["task_to_preds"][lang_goal].append(pred_vec.detach().cpu())
                metrics["task_to_frames"][lang_goal].append(frame_preds.clone())
                vec_np = pred_vec.detach().cpu().numpy()
                base = os.path.splitext(video_basename)[0]
                tsne_video(vec_np, os.path.join(output_dir, f"{base}_tsne.mp4"))
                latent_speed_curve(vec_np, os.path.join(output_dir, f"{base}_speed.png"), delta_t)
                feature_heatmap(vec_np, os.path.join(output_dir, f"{base}_heatmap.png"))
                principal_component_ribbon(vec_np, os.path.join(output_dir, f"{base}_pca.png"), delta_t)
                tsne_trajectory_plot(pred_vec.detach().cpu(), base, os.path.join(output_dir, f"{base}_tsne_traj.png"))

        basename = os.path.basename(video_path).split(".")[0]
        visualization(
            lang_goal=lang_goal,
            orig_video=frames.detach().cpu(),
            decoding_mode2preds=decoding_mode2preds,
            image_seq_post_processor=image_seq_post_processor,
            path=os.path.join(output_dir, basename)
        )

    return metrics



def main(args):
    # Prepare Moto-GPT
    print(f"loading Moto-GPT from {args.moto_gpt_path} ...")
    moto_gpt = load_model(args.moto_gpt_path)
    moto_gpt_config = moto_gpt.config
    moto_gpt = moto_gpt.cuda()
    moto_gpt.eval()

    # Prepare tokenizers and processors
    lang_tokenizer = AutoTokenizer.from_pretrained(moto_gpt_config['model_lang']['pretrained_model_name_or_path'])
    vision_processor_config = json.load(open(get_file_from_repo(moto_gpt_config['model_vision']['pretrained_model_name_or_path'], FEATURE_EXTRACTOR_NAME)))
    image_processor = get_image_processor(vision_processor_config)
    image_seq_post_processor = partial(
        get_image_seq_post_processor, 
        image_std=vision_processor_config['image_std'], 
        image_mean=vision_processor_config['image_mean']
    )

    # Prepare Latent Motion Tokenizer
    print(f"loading Latent Motion Tokenizer from {args.latent_motion_tokenizer_path} ...")
    latent_motion_tokenizer = load_model(args.latent_motion_tokenizer_path)
    latent_motion_tokenizer = latent_motion_tokenizer.cuda()
    latent_motion_tokenizer.eval()


    # Run inference
    metrics = inference(
        moto_gpt=moto_gpt,
        latent_motion_tokenizer=latent_motion_tokenizer,
        lang_tokenizer=lang_tokenizer,
        image_processor=image_processor,
        image_seq_post_processor=image_seq_post_processor,
        delta_t=args.delta_t,
        moto_gpt_seq_len=moto_gpt_config['sequence_length'],
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

    for task, rmses in metrics["task_to_rmses"].items():
        rmse_tensor = torch.cat(rmses)
        print(f"RMSE per step for {task}:", rmse_tensor)
        avg_rmse = rmse_tensor.mean().item()
        print(f"Average RMSE for {task}: {avg_rmse:.6f}")

    tsne_cluster_plot(
        metrics["task_to_preds"],
        os.path.join(args.output_dir, "tsne_cluster_plots"),
    )
    subtrajectory_faiss_analysis(
        metrics["task_to_preds"],
        metrics["task_to_frames"],
        os.path.join(args.output_dir, "subtrajectory_matches"),
        image_seq_post_processor,
    )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--moto_gpt_path', type=str, required=True)
    parser.add_argument('--latent_motion_tokenizer_path', type=str, required=True)
    parser.add_argument('--delta_t', type=int, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    main(args)


import pyrootutils
pyrootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

import argparse
import os
import faiss
import numpy as np
import torch
import omegaconf
import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm

from common.data.data_utils import load_dataset
from common.models.model_utils import load_model

def get_contrastive_mlp(moto_gpt_model):
    return moto_gpt_model.contrastive_mlp

def build_retrieval_index(cfg):
    # Load MotoGPT model to get the contrastive MLP
    moto_gpt_model = load_model(cfg['moto_gpt_path']).to(cfg['device'])
    moto_gpt_model.eval()
    contrastive_mlp = get_contrastive_mlp(moto_gpt_model)

    # Load dataset
    dataset_config_path = cfg['dataset_config_path']
    extra_data_config = {
        'sequence_length': cfg['sequence_length'],
        'latent_motion_pred': True,
        'act_pred': False,
    }
    train_dataset, _ = load_dataset(dataset_config_path, extra_data_config)
    
    # Dataloader
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=cfg['num_workers'])
    
    # Load Latent Motion Tokenizer
    latent_motion_tokenizer = load_model(cfg['latent_motion_tokenizer_path']).to(cfg['device'])
    latent_motion_tokenizer.eval()
    
    all_projected_embeddings = []
    all_latent_vectors = []

    # Extract embeddings for every single timestep
    for i, batch in enumerate(tqdm(dataloader, desc="Extracting embeddings")):
        with torch.no_grad():
            rgb_seq = torch.cat([batch['rgb_initial'], batch['rgb_future']], dim=1)
            rgb_seq = rgb_seq.to(cfg['device'])
            
            b, t, c, h, w = rgb_seq.shape
            t = t - 1
            
            cond_pixel_values = rgb_seq[:, :-1].reshape(-1, c, h, w)
            target_pixel_values = rgb_seq[:, 1:].reshape(-1, c, h, w)
            
            # Get the continuous embeddings from the codebook
            latent_motion_vectors, latent_motion_ids, _ = latent_motion_tokenizer.tokenize(
                cond_pixel_values=cond_pixel_values,
                target_pixel_values=target_pixel_values
            )
            latent_motion_vectors = latent_motion_vectors.mean(dim=1) # Average pooling over motion tokens
            
            # Project into the meaningful embedding space for searching
            projected_embeddings = contrastive_mlp(latent_motion_vectors)
            
            all_projected_embeddings.append(projected_embeddings.cpu().numpy())
            all_latent_vectors.append(latent_motion_vectors.cpu().numpy())

    all_projected_embeddings = np.concatenate(all_projected_embeddings, axis=0).astype(np.float32)
    all_latent_vectors = np.concatenate(all_latent_vectors, axis=0).astype(np.float32)
    
    # Build a single FAISS index on the projected embeddings
    faiss.normalize_L2(all_projected_embeddings)
    index = faiss.IndexFlatIP(all_projected_embeddings.shape[1])
    index.add(all_projected_embeddings)

    # Save index and the corresponding raw latent vectors
    save_path = cfg['save_path']
    os.makedirs(save_path, exist_ok=True)
    faiss.write_index(index, os.path.join(save_path, 'retrieval_index.faiss'))
    np.save(os.path.join(save_path, 'latent_vectors.npy'), all_latent_vectors)
    
    print(f"FAISS index and data mapping built and saved to {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = omegaconf.OmegaConf.load(args.config)
    build_retrieval_index(config) 
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
    torch.cuda.empty_cache()
    moto_gpt_model = load_model(cfg['moto_gpt_path']).to(cfg['device'])
    data_config = omegaconf.OmegaConf.load(cfg.dataset_config_path)
    train_dataset, _ = load_dataset(data_config, {})
    dataset = train_dataset
    batch_size = cfg['batch_size']
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cfg['num_workers'])
    all_action_embeddings = []
    all_actions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating action embeddings"):
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(cfg['device'])
            action_embeddings = moto_gpt_model.get_action_embedding(
                rgb_static=batch['rgb_static'],
                rgb_gripper=batch['rgb_gripper'],
                actions=batch['actions']
            )
            all_action_embeddings.append(action_embeddings.cpu().numpy())
            all_actions.append(batch['actions'].cpu().numpy())
    all_action_embeddings = np.concatenate(all_action_embeddings, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    d = all_action_embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(all_action_embeddings)
    faiss.write_index(index, os.path.join(cfg['retrieval_save_path'], 'retrieval_action_embedding_index.faiss'))
    np.save(os.path.join(cfg['retrieval_save_path'], 'retrieval_action_vectors.npy'), all_actions)
    print("Retrieval index and action vectors saved successfully.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    config = omegaconf.OmegaConf.load(args.config)
    build_retrieval_index(config)
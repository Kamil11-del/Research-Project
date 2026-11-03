"""
ActMamba — main entry point.
Adapted from ActFusion's main.py; all logic is identical.
"""
import os
import torch
import argparse
import numpy as np

from src.dataset import get_data_dict, VideoFeatureDataset
from src.utils import load_config_file, read_mapping_dict
from src.trainer import Trainer

import wandb
import random
from torch.backends import cudnn


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config',     type=str)
    parser.add_argument('--split',      type=int, default=1)
    parser.add_argument('--test',       action='store_true', help='Only run test/inference')
    parser.add_argument('--result_dir', type=str, default='actmamba')
    parser.add_argument('--ckpt',       action='store_true', help='Inference with checkpoint models')
    args = parser.parse_args()

    all_params = load_config_file(args.config)
    locals().update(all_params)

    # Propagate runtime config to args
    args.pos        = all_params.get('pos',        'none')
    args.n_mask     = all_params.get('n_mask',     10)
    args.patch_size = all_params.get('patch_size', 10)

    # naming comes from CLI --result_dir (overrides JSON 'naming' field)
    naming      = args.result_dir
    result_base = all_params.get('result_dir', 'result')
    os.makedirs(result_base, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = all_params['dataset_name']
    project_map  = {
        '50salads': '50s_actmamba',
        'gtea':     'gtea_actmamba',
    }
    wandb.init(project=project_map.get(dataset_name, 'bf_actmamba'))
    wandb.run.name = args.result_dir
    wandb.config.update(vars(args),    allow_val_change=True)
    wandb.config.update(all_params,    allow_val_change=True)

    # ── Data paths ────────────────────────────────────────────────────────
    feature_dir  = os.path.join(root_data_dir, dataset_name, 'features')
    label_dir    = os.path.join(root_data_dir, dataset_name, 'groundTruth')
    mapping_file = os.path.join(root_data_dir, dataset_name, 'mapping.txt')

    actions_dict = read_mapping_dict(mapping_file)
    event_list   = [i[1] for i in np.loadtxt(mapping_file, dtype=str)]
    num_classes  = len(event_list)
    split        = args.split
    print("split:", split)

    train_video_list = [i.split('.')[0] for i in np.loadtxt(
        os.path.join(root_data_dir, dataset_name, 'splits', f'train.split{split}.bundle'), dtype=str)]
    test_video_list  = [i.split('.')[0] for i in np.loadtxt(
        os.path.join(root_data_dir, dataset_name, 'splits', f'test.split{split}.bundle'),  dtype=str)]

    if not args.test:
        train_data_dict = get_data_dict(
            feature_dir=feature_dir, label_dir=label_dir,
            video_list=train_video_list, event_list=event_list,
            sample_rate=sample_rate, temporal_aug=temporal_aug,
            boundary_smooth=boundary_smooth,
        )
        train_train_dataset = VideoFeatureDataset(train_data_dict, num_classes, mode='train')
        train_test_dataset  = VideoFeatureDataset(train_data_dict, num_classes, mode='test')

    test_data_dict = get_data_dict(
        feature_dir=feature_dir, label_dir=label_dir,
        video_list=test_video_list, event_list=event_list,
        sample_rate=sample_rate, temporal_aug=temporal_aug,
        boundary_smooth=boundary_smooth,
    )
    test_test_dataset = VideoFeatureDataset(test_data_dict, num_classes, mode='test')

    trainer = Trainer(
        dict(encoder_params), dict(decoder_params), dict(diffusion_params),
        event_list, sample_rate, temporal_aug, set_sampling_seed, postprocess,
        device=device, args=args,
    )



    if args.test:
        mode = 'decoder-agg'

        if args.ckpt:
            model_path = os.path.join('ckpt', dataset_name, f'split{args.split}.model')
        else:
            model_path = os.path.join('result', args.result_dir, dataset_name,
                                      f'split{args.split}', 'best_combined_model.pth')
        print("model loaded:", model_path)
        result_path = os.path.join('result', args.result_dir, dataset_name, f'split{args.split}')

        # TAS inference
        print("TAS inference")
        trainer.test(test_test_dataset, mode, device, label_dir,
                     result_dir=result_path, model_path=model_path,
                     args=args, all_params=all_params, obs_p=1.0)

        # LTA inference
        for obs_p in [0.2, 0.3]:
            print(f"LTA inference: obs_p={obs_p}")
            trainer.test(test_test_dataset, mode, device, label_dir,
                         result_dir=result_path, model_path=model_path,
                         args=args, all_params=all_params, obs_p=obs_p)
    else:
        trainer.train(
            train_train_dataset, train_test_dataset, test_test_dataset,
            loss_weights, class_weighting, soft_label,
            num_epochs, batch_size, learning_rate, weight_decay,
            label_dir=label_dir,
            result_dir=os.path.join('result', naming, dataset_name, f'split{split}'),
            log_freq=log_freq, log_train_results=log_train_results,
            args=args, all_params=all_params,
        )

import argparse
import random
import os

import numpy as np
import torch

from datasets import faced_dataset, seedv_dataset, physio_dataset, shu_dataset, isruc_dataset, chb_dataset, \
    speech_dataset, mumtaz_dataset, seedvig_dataset, stress_dataset, tuev_dataset, tuab_dataset, bciciv2a_dataset, bciciv2b_dataset
from finetune_trainer import Trainer
from models import model_for_faced, model_for_seedv, model_for_physio, model_for_shu, model_for_isruc, model_for_chb, \
    model_for_speech, model_for_mumtaz, model_for_seedvig, model_for_stress, model_for_tuev, model_for_tuab, \
    model_for_bciciv2a, model_for_bciciv2b, model_for_bciciv2b_adapter


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    parser.add_argument('--use_adapter', type=lambda x: x.lower() == 'true', default=False,
                        help='True, False')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='BCIC-IV-2b',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a]')
    parser.add_argument('--datasets_dir', type=str,
                        default='datasets/downstream/lmdb_2b_0_38Hz',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='CBraMod/downstream_checkpoints', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=lambda x: x.lower() == 'true',
                        default=True, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=lambda x: x.lower() == 'true',
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='CBraMod/pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    print(params)

    setup_seed(params.seed)
    torch.cuda.set_device(params.cuda)
    print('The downstream dataset is {}'.format(params.downstream_dataset))
    
    # ROOT directory (chưa có LOSO)
    root_dir = params.datasets_dir
    # LOOP LOSO

    for i in range(1, 10):
        print("\n" + "="*50)
        print(f"RUN LOSO B0{i}")
        print("="*50)

        # dataset path
        params.datasets_dir = os.path.join(root_dir, f"LOSO_B0{i}")

        # model save path
        params.model_dir = os.path.join(
            'CBraMod/downstream_checkpoints',
            f"LOSO_B0{i}"
        )
        print("Dataset dir:", params.datasets_dir)

        # load data
        load_dataset = bciciv2b_dataset.LoadDataset(params)
        data_loader = load_dataset.get_data_loader()

        # model
        model = model_for_bciciv2b_adapter.Model(params)

        # trainer
        t = Trainer(params, data_loader, model)

        # train
        t.train_for_multiclass()

    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    main()
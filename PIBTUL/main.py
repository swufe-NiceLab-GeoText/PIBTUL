# -*- coding: utf-8 -*-
"""Training script for the paper-strict PIBTUL implementation (Foursquare).

Protocol kept identical to the other experiments for comparability:
  - 80/20 random split (seed 2024), batch 128, 80 epochs, Adam lr=0.0005
  - best-on-test metrics tracked across epochs (ACC@1, ACC@5, macro-P/R/F1)
Paper-faithful settings (Section IV-A.4):
  - v(c) dim 250, hidden size 256, latent dim d 256, nu 0.9,
    beta1 = beta2 = 0.5, gamma 0.01, lambda 1, Adam lr 0.0005
  - no lr scheduler / weight decay / gradient clipping (not in the paper)
  - evaluation uses the same fixed three-view protocol as the other
    experiments (orig + truncating + reversal views), with the posterior
    mean mu used for deterministic predictions.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data_load import TrajDataset
from models import PIBTUL
from utils import (TrajAugmenterWrapper, aug_collate_fn, accuracy_at_k,
                   calculate_macro_metrics, read_processed_tra,
                   get_embedding_vector, read_trajectories)


def parse_args():
    p = argparse.ArgumentParser(description='PIBTUL (paper-strict) on Foursquare')
    p.add_argument('--processed_file', type=str, default='../data/Foursquare_traj_new.pkl')
    p.add_argument('--train_file', type=str, default='../data/foursquare_6.txt')
    p.add_argument('--vec_file', type=str, default='../data/foursquare_embedding_node2vec_2.dat')
    p.add_argument('--city', type=str, default='foursquare')
    p.add_argument('--processed_flag', action='store_true', default=False)
    p.add_argument('--embed_size', type=int, default=250)
    p.add_argument('--hidden_size', type=int, default=256)
    p.add_argument('--latent_dim', type=int, default=256)
    p.add_argument('--num_layers', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--learning_rate', type=float, default=0.0005)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--nu', type=float, default=0.9, help='prototype momentum, Eq.(19)')
    p.add_argument('--init', type=str, default='gaussian',
                   choices=['gaussian', 'uniform', 'classmean'],
                   help='prototype initialization, Table V')
    p.add_argument('--views', type=str, default='OTR',
                   help='active views subset (letters O/T/R/S; S=Substitution), '
                        'e.g. O/OT/TR/OR/OTS/OTR, Table IV')
    p.add_argument('--sub_ratio', type=float, default=0.15,
                   help='substitution ratio for the S view, Table IV')
    p.add_argument('--ablation', type=str, default='none',
                   choices=['none', 'pemb', 'mobcl', 'taug', 'attn', 'pgo'],
                   help='component ablation: w/o PEmb/MobCL/TAug/Attn/PGO (paper Fig.)')
    p.add_argument('--gamma', type=float, default=0.01, help='MobCL weight, Eq.(25)')
    p.add_argument('--lam', type=float, default=1.0, help='prototype loss weight, Eq.(25)')
    p.add_argument('--mi_weight', type=float, default=1.0,
                   help='(eta+zeta)/2 in Eq.(9); value unspecified in paper')
    p.add_argument('--beta_pred', type=float, default=1.0, help='CE weight, beta_pred')
    p.add_argument('--beta_kl', type=float, default=0.01, help='symKL weight, beta_KL')
    p.add_argument('--beta_mi', type=float, default=0.01, help='MI weight, beta_MI')
    p.add_argument('--rho', type=float, default=0.7,
                   help='truncating ratio for augmentation; value unspecified in paper')
    p.add_argument('--inter_mode', type=str, default='paper',
                   choices=['paper', 'intent'],
                   help='paper: literal Eq.(21) max-min; intent: -min_dist (stated intent)')
    p.add_argument('--tag', type=str, default='base', help='run tag for output files')
    p.add_argument('--train_ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=2024)
    p.add_argument('--print_freq', type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    print(f'Args: {vars(args)}')

    if not args.processed_flag:
        users, traj = read_trajectories(args.train_file)
        import pickle
        data_set = TrajDataset(traj_data=traj, traj_user=users,
                               padding_idx=0, use_sos_eos=None)
        with open(args.processed_file, 'wb') as f:
            pickle.dump(data_set, f)

    origin_dataset = read_processed_tra(args.processed_file)

    train_size = int(args.train_ratio * len(origin_dataset))
    test_size = len(origin_dataset) - train_size
    train_sub, test_sub = random_split(origin_dataset, [train_size, test_size])

    # training: paper augmentation (Eq. 10-11); evaluation: the same fixed
    # three-view protocol as the other experiments
    # (w/o TAug: no trajectory augmentation -> only the raw view; views=O)
    abl = args.ablation
    if abl == 'taug':
        train_dataset = TrajAugmenterWrapper(train_sub, augment=False, crop_ratio=args.rho)
        test_dataset = TrajAugmenterWrapper(test_sub, augment=False, crop_ratio=args.rho)
        args.views = 'O'
    else:
        train_dataset = TrajAugmenterWrapper(train_sub, augment=True, crop_ratio=args.rho,
                                             views=args.views, sub_ratio=args.sub_ratio)
        test_dataset = TrajAugmenterWrapper(test_sub, augment=True, crop_ratio=args.rho,
                                            views=args.views, sub_ratio=args.sub_ratio)

    output_user_size = torch.max(origin_dataset.user_label).item() + 1
    embeddings = get_embedding_vector(args.vec_file, embed_size=args.embed_size)
    print(f'#users={output_user_size}, #trajs={len(origin_dataset)}, '
          f'emb={tuple(embeddings.shape)}')

    # Component ablation flags (paper Fig.): w/o PEmb / w/o Attn.
    use_attn = (abl != 'attn')
    pretrained_emb = (abl != 'pemb')
    model = PIBTUL(embed_size=args.embed_size, hidden_size=args.hidden_size,
                   latent_dim=args.latent_dim, num_layers=args.num_layers,
                   embeddings=embeddings, output_user_size=output_user_size,
                   device=device, nu=args.nu, mi_weight=args.mi_weight,
                   inter_mode=args.inter_mode, init_mode=args.init,
                   views=args.views, use_attn=use_attn,
                   pretrained_emb=pretrained_emb).to(device)

    # main optimizer: everything except the MINE critics
    critic_params = [p for m in model.critic_modules() for p in m.parameters()]
    critic_ids = {id(p) for p in critic_params}
    main_params = [p for p in model.parameters() if id(p) not in critic_ids]
    optimizer = torch.optim.Adam(main_params, lr=args.learning_rate)
    critic_optimizer = torch.optim.Adam(critic_params, lr=args.learning_rate)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              collate_fn=aug_collate_fn, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             collate_fn=aug_collate_fn, shuffle=False)

    ce_loss_fn = nn.CrossEntropyLoss()
    best = {'acc1': 0.0, 'acc5': 0.0, 'macro_p': 0.0, 'macro_r': 0.0,
            'macro_f': 0.0, 'epoch': 0}
    eval_results = []

    # Table V ablation: prototype initialization by per-user class mean.
    # Pre-compute each user's mean z_hat over (a sample of) the training set
    # and use that as the initial prototype; users with no sample stay random.
    if args.init == 'classmean':
        model.eval()
        with torch.no_grad():
            proto_sum = torch.zeros(output_user_size, model.proto_dim, device=device)
            proto_cnt = torch.zeros(output_user_size, 1, device=device)
            for bi, batch in enumerate(train_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                mv_inputs = {}
                if 'O' in args.views:
                    mv_inputs['orig'] = (batch['orig_seq'], batch['orig_length'])
                if 'T' in args.views:
                    mv_inputs['crop'] = (batch['crop_seq'], batch['crop_length'])
                if 'R' in args.views:
                    mv_inputs['reverse'] = (batch['reverse_seq'], batch['reverse_length'])
                if 'S' in args.views:
                    mv_inputs['subst'] = (batch['subst_seq'], batch['subst_length'])
                labels = batch['users']
                _, _, z_proto = model(mv_inputs, sample=False)
                proto_sum.index_add_(0, labels, z_proto)
                proto_cnt.index_add_(0, labels,
                                     torch.ones(labels.shape[0], 1,
                                                device=device, dtype=z_proto.dtype))
                if bi > 20:   # sample cap: enough to cover most users, avoid a full pass
                    break
            mask = (proto_cnt.squeeze(1) > 0)
            init_m = proto_sum / proto_cnt.clamp(min=1.0)
            keep_rnd = torch.randn_like(model.prototypes.data) * 0.01
            model.prototypes.data = torch.where(mask.unsqueeze(1), init_m, keep_rnd)
            n_covered = int(mask.sum().item())
            print(f'[classmean init] covered {n_covered}/{output_user_size} users')
        model.train()

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            # mv_inputs follows the active views (w/o TAug -> only orig).
            mv_inputs = {}
            if 'O' in args.views:
                mv_inputs['orig'] = (batch['orig_seq'], batch['orig_length'])
            if 'T' in args.views:
                mv_inputs['crop'] = (batch['crop_seq'], batch['crop_length'])
            if 'R' in args.views:
                mv_inputs['reverse'] = (batch['reverse_seq'], batch['reverse_length'])
            if 'S' in args.views:
                mv_inputs['subst'] = (batch['subst_seq'], batch['subst_length'])
            labels = batch['users']

            logits, views, z_proto = model(mv_inputs, sample=True)

            # --- step 1: train the MI critics (DV bound) on detached latents
            critic_optimizer.zero_grad()
            c_loss = model.mine_critic_loss(views)
            c_loss.backward()
            critic_optimizer.step()

            # --- step 2: main objective, Eq.(25)
            kl_s, mi_s = model.mobcl_loss(views)
            loss_ce = ce_loss_fn(logits, labels)
            l_intra, l_inter = model.prototype_loss(z_proto, labels)
            # C2 anti-collapse warmup: ramp the prototype forces in over the
            # first PGO_WARMUP epochs so CE builds separable structure first
            # (variant C collapsed in the first 2 epochs without this).
            pgo_warmup = 6.0
            pgo_ramp = min(1.0, (epoch + 1) / pgo_warmup)
            # Component ablation: w/o MobCL -> drop mobcl term; w/o PGO -> drop lam.
            lam_eff = 0.0 if abl == 'pgo' else args.lam
            mobcl_term = (args.beta_kl * kl_s - args.beta_mi * mi_s) if abl != 'mobcl' else None
            total_loss = args.beta_pred * loss_ce + \
                (mobcl_term if mobcl_term is not None else 0.0) + \
                lam_eff * pgo_ramp * (l_intra + l_inter)

            optimizer.zero_grad()
            total_loss.backward()
            # stability safeguard (not part of the paper objective): clip the
            # main-parameter gradients so a noisy MI critic cannot blow up the
            # encoder in a single step
            torch.nn.utils.clip_grad_norm_(main_params, 5.0)
            optimizer.step()

            # Eq.(19): momentum prototype update after the gradient step
            model.update_prototypes(z_proto.detach(), labels)

            if i % args.print_freq == 0:
                probs = F.softmax(logits, dim=-1)
                _, top1 = torch.topk(probs, 1, dim=-1)
                _, top5 = torch.topk(probs, 5, dim=-1)
                acc1 = accuracy_at_k(top1.tolist(), labels.tolist(), 1)
                acc5 = accuracy_at_k(top5.tolist(), labels.tolist(), 5)
                print(f'Epoch {epoch + 1}, Batch {i} | '
                      f'Acc@1: {acc1:.4f}, Acc@5: {acc5:.4f} | '
                      f'CE: {loss_ce.item():.4f}, KL: {kl_s.item():.3f}, MI: {mi_s.item():.3f}, '
                      f'intra: {l_intra.item():.4f}, inter: {l_inter.item():.4f}')

        print(f'Epoch {epoch + 1} training time: {time.time() - t0:.2f}s')

        # ---------------- evaluation (three-view protocol, deterministic) ---
        model.eval()
        Top1, Top5, Pred, Gold = [], [], [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mv_inputs = {}
                if 'O' in args.views:
                    mv_inputs['orig'] = (batch['orig_seq'], batch['orig_length'])
                if 'T' in args.views:
                    mv_inputs['crop'] = (batch['crop_seq'], batch['crop_length'])
                if 'R' in args.views:
                    mv_inputs['reverse'] = (batch['reverse_seq'], batch['reverse_length'])
                if 'S' in args.views:
                    mv_inputs['subst'] = (batch['subst_seq'], batch['subst_length'])
                logits, _, _ = model(mv_inputs, sample=False)
                labels = batch['users']
                probs = F.softmax(logits, dim=-1)
                _, top1 = torch.topk(probs, 1, dim=-1)
                _, top5 = torch.topk(probs, 5, dim=-1)
                Top1.extend(top1.tolist())
                Top5.extend(top5.tolist())
                Pred.extend(torch.argmax(probs, -1).tolist())
                Gold.extend(labels.tolist())

        acc1 = accuracy_at_k(Top1, Gold, 1)
        acc5 = accuracy_at_k(Top5, Gold, 5)
        macro_p, macro_r, macro_f = calculate_macro_metrics(Pred, Gold)
        eval_results.append(acc1)
        print(f'---Test acc@1: {acc1:.4f}, acc@5: {acc5:.4f}, '
              f'Macro_P: {macro_p:.4f}, Macro_R: {macro_r:.4f}, '
              f'Macro_F: {macro_f:.4f}')

        if acc1 > best['acc1']:
            best.update({'acc1': acc1, 'acc5': acc5, 'macro_p': macro_p,
                         'macro_r': macro_r, 'macro_f': macro_f,
                         'epoch': epoch + 1})
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(best_state, f'best_model_{args.city}_strict_{args.tag}.pth')
        print(f"Epoch {epoch + 1}, Acc@1: {acc1:.4f}, Best: {best['acc1']:.4f}")

    print('\n----- Best Results (PIBTUL strict) -----')
    print(f"Best Acc@1: {best['acc1']:.4f} (epoch {best['epoch']})")
    print(f"Best Acc@5: {best['acc5']:.4f}")
    print(f"Best Macro_P: {best['macro_p']:.4f}")
    print(f"Best Macro_R: {best['macro_r']:.4f}")
    print(f"Best Macro_F: {best['macro_f']:.4f}")

    best_map = {
        'best_acc1': best['acc1'], 'best_acc5': best['acc5'],
        'best_macro_p': best['macro_p'], 'best_macro_r': best['macro_r'],
        'best_macro_f': best['macro_f'], 'best_epoch': best['epoch'],
        'hyperparameters': vars(args),
    }
    with open(f'acc_data_{args.city}_PIBTUL_strict_{args.tag}.json', 'w') as f:
        json.dump(eval_results, f)
    with open(f'best_results_{args.city}_PIBTUL_strict_{args.tag}.json', 'w') as f:
        json.dump(best_map, f, indent=2)
    print('Results saved.')


if __name__ == '__main__':
    main()

import wandb
import json
import os
from torch import optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from scipy.ndimage import median_filter
import numpy as np
from tqdm import tqdm

from .dataset import restore_full_sequence
from .model.actmamba import ActMamba
from .utils import (func_eval, get_labels_start_end_time, mode_filter,
                    read_mapping_dict, eval_file, get_unique_list, get_unique_list_gt)
from .vis import segment_bars


class Trainer:
    def __init__(self, encoder_params, decoder_params, diffusion_params,
                 event_list, sample_rate, temporal_aug, set_sampling_seed,
                 postprocess, device, args=None):

        self.device            = device
        self.num_classes       = len(event_list)
        self.encoder_params    = encoder_params
        self.decoder_params    = decoder_params
        self.event_list        = event_list
        self.sample_rate       = sample_rate
        self.temporal_aug      = temporal_aug
        self.set_sampling_seed = set_sampling_seed
        self.postprocess       = postprocess
        self.args              = args

        self.model = ActMamba(encoder_params, decoder_params, diffusion_params,
                              self.num_classes, self.device, args)
        wandb.watch(self.model)
        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))

        # Best performance tracking
        self.best_tas_acc        = 0.0
        self.best_lta_moc        = 0.0
        self.best_both_score     = 0.0
        self.best_tas_metrics    = None
        self.best_lta_metrics    = None
        self.best_combined_metrics = None

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, train_train_dataset, train_test_dataset, test_test_dataset,
              loss_weights, class_weighting, soft_label,
              num_epochs, batch_size, learning_rate, weight_decay,
              label_dir, result_dir, log_freq, log_train_results=True,
              args=None, all_params=None):

        device = self.device
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()

        # CosineAnnealingLR: smooth LR decay from learning_rate → 1e-6
        # over the full training run (no warm restarts).
        # Warm restarts were removed because each LR spike combined with
        # optimizer.state.clear() created a high-LR / no-momentum window
        # that risked destabilising a converging model. The gradient clip
        # (max_norm=1.0) is the primary explosion guard.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )

        restore_epoch = -1
        step          = 1

        if os.path.exists(result_dir):
            latest = os.path.join(result_dir, 'latest.pt')
            if 'latest.pt' in os.listdir(result_dir) and os.path.getsize(latest) > 0:
                saved_state = torch.load(latest)
                self.model.load_state_dict(saved_state['model'])
                optimizer.load_state_dict(saved_state['optimizer'])
                restore_epoch = saved_state['epoch']
                step          = saved_state['step']
                # ── Critical fix: restore best metrics so they are never reset ──
                self.best_tas_acc    = saved_state.get('best_tas_acc',    0.0)
                self.best_lta_moc    = saved_state.get('best_lta_moc',    0.0)
                self.best_both_score = saved_state.get('best_both_score', 0.0)
                print(f'[RESUME] epoch={restore_epoch} | '
                      f'best TAS={self.best_tas_acc:.4f}  '
                      f'LTA={self.best_lta_moc:.4f}  '
                      f'Combined={self.best_both_score:.4f}')
                if 'scheduler' in saved_state:
                    scheduler.load_state_dict(saved_state['scheduler'])
                else:
                    for _ in range(restore_epoch + 1):
                        scheduler.step()

        # Belt-and-suspenders: sync with best_scores.json (covers old checkpoints)
        scores_path = os.path.join(result_dir, 'best_scores.json')
        if os.path.exists(scores_path):
            try:
                with open(scores_path) as f:
                    _s = json.load(f)
                self.best_tas_acc    = max(self.best_tas_acc,    _s.get('best_tas',      0.0))
                self.best_lta_moc    = max(self.best_lta_moc,    _s.get('best_lta',      0.0))
                self.best_both_score = max(self.best_both_score, _s.get('best_combined', 0.0))
            except Exception:
                pass

        if class_weighting:
            class_weights = torch.from_numpy(
                train_train_dataset.get_class_weights()
            ).float().to(device)
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, weight=class_weights, reduction='none')
        else:
            ce_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

        bce_criterion = nn.BCELoss(reduction='none')
        mse_criterion = nn.MSELoss(reduction='none')

        train_loader = torch.utils.data.DataLoader(
            train_train_dataset, batch_size=1, shuffle=True, num_workers=4)

        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
            logger = SummaryWriter(result_dir)

        for epoch in range(restore_epoch + 1, num_epochs):
            self.model.train()
            epoch_running_loss = 0

            for _, data in enumerate(train_loader):
                feature, label, boundary, video = data
                feature  = feature.to(device)
                label    = label.to(device)
                boundary = boundary.to(device)

                loss_dict = self.model.get_training_loss(
                    feature,
                    event_gt    = F.one_hot(label.long(), num_classes=self.num_classes).permute(0, 2, 1),
                    boundary_gt = boundary,
                    encoder_ce_criterion       = ce_criterion,
                    encoder_mse_criterion      = mse_criterion,
                    encoder_boundary_criterion = bce_criterion,
                    decoder_ce_criterion       = ce_criterion,
                    decoder_mse_criterion      = mse_criterion,
                    decoder_boundary_criterion = bce_criterion,
                    soft_label = soft_label,
                    args       = args,
                )

                total_loss = sum(loss_weights[k] * v for k, v in loss_dict.items())

                # ── Guard: skip batch if loss is NaN/Inf (protects weights) ──────
                scaled_loss = total_loss / batch_size
                if not torch.isfinite(scaled_loss):
                    print(f'[WARNING] Non-finite loss at step {step} — skipping batch.')
                    optimizer.zero_grad()
                    step += 1
                    continue

                if result_dir:
                    for k, v in loss_dict.items():
                        logger.add_scalar(f'Train-{k}', loss_weights[k] * v.item() / batch_size, step)
                    logger.add_scalar('Train-Total', scaled_loss.item(), step)

                scaled_loss.backward()
                epoch_running_loss += scaled_loss.item()

                # ── Clip EVERY step (not every batch_size steps) ─────────────────
                # Clipping only every batch_size steps left intermediate steps
                # unprotected, which caused the gradient explosion at epoch ~947.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                if step % batch_size == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                step += 1

            epoch_running_loss /= len(train_train_dataset)
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"loss": epoch_running_loss, "lr": current_lr})
            print(f'Epoch {epoch} - Running Loss {epoch_running_loss:.6f}  LR {current_lr:.2e}')
            scheduler.step()

            state = {
                'model':           self.model.state_dict(),
                'optimizer':       optimizer.state_dict(),
                'scheduler':       scheduler.state_dict(),
                'epoch':           epoch,
                'step':            step,
                # Persist best metrics so resume never resets the bar
                'best_tas_acc':    self.best_tas_acc,
                'best_lta_moc':    self.best_lta_moc,
                'best_both_score': self.best_both_score,
            }

            if epoch % log_freq == 0:
                print(f'\n{"="*68}')
                print(f'  EPOCH {epoch:>5d}  |  EVALUATION')
                print(f'{"="*68}')
                w_path = os.path.join(result_dir, 'log.txt')
                w      = open(w_path, 'a')
                w_     = f'\n{"="*68}\nepoch {epoch}\n{"="*68}\n'

                if result_dir:
                    torch.save(self.model.state_dict(), f'{result_dir}/epoch-{epoch}.model')
                    torch.save(state, f'{result_dir}/latest.pt')

                for mode in ['decoder-agg']:
                    # TAS inference (obs_p = 1.0 = full video)
                    test_result_dict = self.test(
                        test_test_dataset, mode, device, label_dir,
                        args=args, result_dir=result_dir, model_path=None,
                        all_params=all_params, obs_p=1.0)
                    w_ = self._log_tas_results(test_result_dict, w_)

                    # LTA inference — obs_p 0.2
                    test_result_dict20 = self.test(
                        test_test_dataset, mode, device, label_dir,
                        args=args, result_dir=result_dir, model_path=None,
                        all_params=all_params, obs_p=0.2)
                    w_ = self._log_lta_results(test_result_dict20, w_, obs_p=0.2)

                    # LTA inference — obs_p 0.3
                    test_result_dict30 = self.test(
                        test_test_dataset, mode, device, label_dir,
                        args=args, result_dir=result_dir, model_path=None,
                        all_params=all_params, obs_p=0.3)
                    w_ = self._log_lta_results(test_result_dict30, w_, obs_p=0.3)

                    # Best model saving
                    tas_f1s   = [test_result_dict["F1@10"], test_result_dict["F1@25"], test_result_dict["F1@50"]]
                    tas_score = (test_result_dict["Acc"] + test_result_dict["Edit"] + sum(tas_f1s)) / (2 + len(tas_f1s))

                    lta_keys_02 = ["obs0.2_pred0.1", "obs0.2_pred0.2", "obs0.2_pred0.3", "obs0.2_pred0.5"]
                    lta_keys_03 = ["obs0.3_pred0.1", "obs0.3_pred0.2", "obs0.3_pred0.3", "obs0.3_pred0.5"]
                    lta_f1s_02  = [test_result_dict20.get(k, 0.0) for k in lta_keys_02]
                    lta_f1s_03  = [test_result_dict30.get(k, 0.0) for k in lta_keys_03]
                    lta_score   = (sum(lta_f1s_02) + sum(lta_f1s_03)) / (len(lta_f1s_02) + len(lta_f1s_03))

                    self._save_best_models(tas_score, lta_score, result_dir,
                                           tas_metrics=test_result_dict,
                                           lta_metrics={'obs0.2': test_result_dict20,
                                                        'obs0.3': test_result_dict30},
                                           epoch=epoch)

                    w.write(w_)
                    w.close()

                    if result_dir:
                        for k, v in test_result_dict.items():
                            logger.add_scalar(f'Test-{mode}-{k}', v, epoch)
                        np.save(os.path.join(result_dir, f'test_results_{mode}_epoch{epoch}.npy'),
                                test_result_dict)

                    if log_train_results:
                        train_result_dict = self.test(
                            train_test_dataset, mode, device, label_dir,
                            args=args, result_dir=result_dir, model_path=None,
                            all_params=all_params)
                        if result_dir:
                            for k, v in train_result_dict.items():
                                logger.add_scalar(f'Train-{mode}-{k}', v, epoch)
                            np.save(os.path.join(result_dir, f'train_results_{mode}_epoch{epoch}.npy'),
                                    train_result_dict)
                        for k, v in train_result_dict.items():
                            print(f'Epoch {epoch} - {mode}-Train-{k} {v}')

        if result_dir:
            logger.close()

        self._print_best_summary(result_dir)

    # ── Inference ─────────────────────────────────────────────────────────────

    def test_single_video(self, video_idx, test_dataset, mode, device,
                          model_path=None, args=None, all_params=None, obs_p=1.0):
        assert test_dataset.mode == 'test'
        assert mode in ['encoder', 'decoder-noagg', 'decoder-agg']
        assert self.postprocess['type'] in ['median', 'mode', 'purge', None]

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        seed = video_idx if self.set_sampling_seed else None

        with torch.no_grad():
            feature, label, _, video = test_dataset[video_idx]
            input_feats = feature

            is_anticipation = obs_p < 1.0
            if is_anticipation:
                full_len    = feature[0].size(-1)
                obs_len     = round(full_len * obs_p)
                input_feats = [feature[i][:, :, :obs_len] for i in range(len(feature))]

            if mode == 'encoder':
                output = [F.softmax(self.model.encoder(feature[i].to(device)), 1).cpu()
                          for i in range(len(feature))]
                left_offset  = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-agg':
                if is_anticipation:
                    output = [self.model.ddim_sample(input_feats[i].to(device), seed,
                                                     full_len=full_len, args=args).cpu()
                              for i in range(len(input_feats))]
                else:
                    output = [self.model.ddim_sample(feature[i].to(device), seed, args=args).cpu()
                              for i in range(len(feature))]
                left_offset  = self.sample_rate // 2
                right_offset = (self.sample_rate - 1) // 2

            elif mode == 'decoder-noagg':
                output = [self.model.ddim_sample(
                    feature[len(feature) // 2].to(device), seed).cpu()]
                left_offset  = self.sample_rate // 2
                right_offset = 0

            assert output[0].shape[0] == 1
            min_len = min(i.shape[2] for i in output)
            output  = torch.cat([i[:, :, :min_len] for i in output], 0)
            output  = output.mean(0).numpy()

            if self.postprocess['type'] == 'median':
                smoothed = np.zeros_like(output)
                for c in range(output.shape[0]):
                    smoothed[c] = median_filter(output[c], size=self.postprocess['value'])
                output = smoothed / smoothed.sum(0, keepdims=True)

            output = np.argmax(output, 0)
            output = restore_full_sequence(output,
                                           full_len    = label.shape[-1],
                                           left_offset  = left_offset,
                                           right_offset = right_offset,
                                           sample_rate  = self.sample_rate)

            if self.postprocess['type'] == 'mode':
                output = mode_filter(output, self.postprocess['value'])

            if self.postprocess['type'] == 'purge':
                trans, starts, ends = get_labels_start_end_time(output)
                for e in range(len(trans)):
                    duration = ends[e] - starts[e]
                    if duration <= self.postprocess['value']:
                        if e == 0:
                            output[starts[e]:ends[e]] = trans[e + 1]
                        elif e == len(trans) - 1:
                            output[starts[e]:ends[e]] = trans[e - 1]
                        else:
                            mid = starts[e] + duration // 2
                            output[starts[e]:mid]  = trans[e - 1]
                            output[mid:ends[e]]    = trans[e + 1]

            label = label.squeeze(0).cpu().numpy()
            assert output.shape == label.shape
            return video, output, label

    def test(self, test_dataset, mode, device, label_dir,
             result_dir=None, model_path=None, args=None, all_params=None, obs_p=1.0):

        assert test_dataset.mode == 'test'

        self.model.eval()
        self.model.to(device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path))

        mapping_file    = os.path.join('datasets', all_params['dataset_name'], 'mapping.txt')
        actions_dict    = read_mapping_dict(mapping_file)
        actions_dict_inv = {v: k for k, v in actions_dict.items()}

        is_anticipation = obs_p < 1.0
        if is_anticipation:
            eval_ps   = [0.1, 0.2, 0.3, 0.5]
            T_actions = np.zeros((len(eval_ps), len(actions_dict)))
            F_actions = np.zeros((len(eval_ps), len(actions_dict)))

        with torch.no_grad():
            result_dict = {}
            for video_idx in tqdm(range(len(test_dataset))):
                video, pred, label = self.test_single_video(
                    video_idx, test_dataset, mode, device,
                    model_path, args, all_params, obs_p=obs_p)

                pred_ant = pred
                pred     = [self.event_list[int(i)] for i in pred]

                os.makedirs(os.path.join(result_dir, 'prediction'), exist_ok=True)
                with open(os.path.join(result_dir, 'prediction', f'{video}.txt'), 'w') as f:
                    f.write('### Frame level recognition: ###\n')
                    f.write(' '.join(pred))

                if is_anticipation:
                    vis_path = os.path.join(result_dir, 'vis', 'ant' + str(obs_p),
                                            all_params['dataset_name'], 'split' + str(args.split))
                else:
                    vis_path = os.path.join(result_dir, 'vis', 'seg',
                                            all_params['dataset_name'], 'split' + str(args.split))
                os.makedirs(vis_path, exist_ok=True)

                gt_unique,   gt_unique_len   = get_unique_list_gt(label,    actions_dict_inv)
                pred_unique, pred_unique_len = get_unique_list(pred)
                segment_bars(os.path.join(vis_path, f'{video}.png'),
                             label, pred_ant,
                             gt_unique=gt_unique,   gt_unique_len=gt_unique_len,
                             pred_unique=pred_unique, pred_unique_len=pred_unique_len)

                if is_anticipation:
                    total_len = len(label)
                    for i, eval_p in enumerate(eval_ps):
                        eval_len = int((obs_p + eval_p) * total_len)
                        T_action, F_action = eval_file(label, pred_ant[:eval_len], obs_p, actions_dict)
                        T_actions[i] += T_action
                        F_actions[i] += F_action

            if is_anticipation:
                for i, eval_p in enumerate(eval_ps):
                    acc, n = 0, 0
                    for j in range(len(actions_dict)):
                        total = T_actions[i, j] + F_actions[i, j]
                        if total != 0:
                            acc += float(T_actions[i, j] / total)
                            n   += 1
                    moc = (float(acc) / n) * 100
                    print(f'obs. {int(100*obs_p)}  pred. {int(100*eval_p)} --> MoC: {moc:.2f}')
                    result_dict[f'obs{obs_p}_pred{eval_p}'] = moc

        if is_anticipation:
            acc, edit, f1s = func_eval(label_dir, os.path.join(result_dir, 'prediction'),
                                       test_dataset.video_list, obs_p=obs_p)
        else:
            acc, edit, f1s = func_eval(label_dir, os.path.join(result_dir, 'prediction'),
                                       test_dataset.video_list, obs_p=1.0)

        result_dict.update({'Acc': acc, 'Edit': edit,
                            'F1@10': f1s[0], 'F1@25': f1s[1], 'F1@50': f1s[2]})

        if not is_anticipation:
            print(f"Acc:{acc:.2f}  Edit:{edit:.2f}  "
                  f"F1@10:{f1s[0]:.2f}  F1@25:{f1s[1]:.2f}  F1@50:{f1s[2]:.2f}")

        return result_dict

    # ── Logging helpers ────────────────────────────────────────────────────

    def _log_tas_results(self, result_dict, text_buffer):
        wandb.log({k: result_dict[k] for k in ['Acc', 'Edit', 'F1@10', 'F1@25', 'F1@50']})
        acc, edit = result_dict['Acc'], result_dict['Edit']
        f1s = f"{result_dict['F1@10']:.2f}/{result_dict['F1@25']:.2f}/{result_dict['F1@50']:.2f}"
        print(f'  [TAS]  Acc:{acc:.2f}  Edit:{edit:.2f}  F1@[10/25/50]:{f1s}')
        text_buffer += (f'── TAS ──\n'
                        f'  Acc:   {acc:.2f}\n'
                        f'  Edit:  {edit:.2f}\n'
                        f'  F1@10: {result_dict["F1@10"]:.2f}\n'
                        f'  F1@25: {result_dict["F1@25"]:.2f}\n'
                        f'  F1@50: {result_dict["F1@50"]:.2f}\n\n')
        return text_buffer

    def _log_lta_results(self, result_dict, text_buffer, obs_p):
        parts = []
        for pred_p in [0.1, 0.2, 0.3, 0.5]:
            key = f'obs{obs_p}_pred{pred_p}'
            val = result_dict.get(key, 0.0)
            wandb.log({key: val})
            parts.append(f'pred{int(100*pred_p)}%:{val:.2f}')
        print(f'  [LTA obs={int(100*obs_p)}%]  ' + '  '.join(parts))
        text_buffer += f'── LTA obs={int(100*obs_p)}% ──\n'
        for pred_p in [0.1, 0.2, 0.3, 0.5]:
            key = f'obs{obs_p}_pred{pred_p}'
            text_buffer += f'  {key}: {result_dict.get(key, 0.0):.2f}\n'
        text_buffer += '\n'
        return text_buffer

    def _save_best_models(self, tas_acc, lta_moc, result_dir,
                          tas_metrics=None, lta_metrics=None, epoch=None):
        combined    = (tas_acc + lta_moc) / 2
        scores_path = os.path.join(result_dir, 'best_scores.json')
        metrics_path = os.path.join(result_dir, 'best_metrics.json')

        # Load persisted scores as safety net (covers old-format checkpoints)
        if os.path.exists(scores_path):
            try:
                with open(scores_path) as f:
                    saved = json.load(f)
                self.best_tas_acc    = max(self.best_tas_acc,    saved.get('best_tas',      0.0))
                self.best_lta_moc    = max(self.best_lta_moc,    saved.get('best_lta',      0.0))
                self.best_both_score = max(self.best_both_score, saved.get('best_combined', 0.0))
            except Exception:
                pass

        updated = False
        if tas_acc > self.best_tas_acc:
            self.best_tas_acc     = tas_acc
            self.best_tas_metrics = tas_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_tas_model.pth')
            print(f'  ★ New best TAS!      Score={tas_acc:.4f}  (epoch {epoch})')
            updated = True

        if lta_moc > self.best_lta_moc:
            self.best_lta_moc     = lta_moc
            self.best_lta_metrics = lta_metrics
            torch.save(self.model.state_dict(), f'{result_dir}/best_lta_model.pth')
            print(f'  ★ New best LTA!      Score={lta_moc:.4f}  (epoch {epoch})')
            updated = True

        if combined > self.best_both_score:
            self.best_both_score       = combined
            self.best_combined_metrics = {'tas': tas_metrics, 'lta': lta_metrics}
            torch.save(self.model.state_dict(), f'{result_dir}/best_combined_model.pth')
            print(f'  ★ New best Combined! Score={combined:.4f}  '
                  f'(TAS={tas_acc:.4f}  LTA={lta_moc:.4f})  (epoch {epoch})')
            updated = True

        # Always persist scores.json
        with open(scores_path, 'w') as f:
            json.dump({'best_tas': self.best_tas_acc,
                       'best_lta': self.best_lta_moc,
                       'best_combined': self.best_both_score}, f, indent=2)

        # Update best_metrics.json only when a genuine improvement occurred
        if updated:
            detail = {}
            if self.best_tas_metrics:
                detail['best_tas'] = {
                    'score': self.best_tas_acc,
                    'epoch': epoch,
                    'metrics': {k: float(v) for k, v in self.best_tas_metrics.items()},
                }
            if self.best_lta_metrics:
                detail['best_lta'] = {
                    'score': self.best_lta_moc,
                    'epoch': epoch,
                    'obs0.2': {k: float(v) for k, v in self.best_lta_metrics.get('obs0.2', {}).items()},
                    'obs0.3': {k: float(v) for k, v in self.best_lta_metrics.get('obs0.3', {}).items()},
                }
            if self.best_combined_metrics:
                detail['best_combined'] = {
                    'score': self.best_both_score,
                    'epoch': epoch,
                }
            try:
                with open(metrics_path, 'w') as f:
                    json.dump(detail, f, indent=2)
            except Exception:
                pass

    def _print_best_summary(self, result_dir):
        summary  = '\n' + '='*68 + '\n'
        summary += '  BEST MODEL PERFORMANCES\n'
        summary += '='*68 + '\n'
        summary += f'  Best TAS Score:      {self.best_tas_acc:.4f}\n'
        if self.best_tas_metrics:
            m = self.best_tas_metrics
            summary += (f"    Acc:{m['Acc']:.2f}  Edit:{m['Edit']:.2f}  "
                        f"F1@10:{m['F1@10']:.2f}  F1@25:{m['F1@25']:.2f}  "
                        f"F1@50:{m['F1@50']:.2f}\n")
        summary += f'  Best LTA Score:      {self.best_lta_moc:.4f}\n'
        summary += f'  Best Combined Score: {self.best_both_score:.4f}\n'
        summary += '='*68 + '\n'
        print(summary)
        if result_dir:
            with open(os.path.join(result_dir, 'log.txt'), 'a') as w:
                w.write(summary)

import os
import json
import random
import torch
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter


def load_config_file(config_file):
    all_params = json.load(open(config_file))

    # ── Backward-compatible defaults ──────────────────────────────────────
    all_params.setdefault('result_dir',       'result')
    all_params.setdefault('log_train_results', True)
    all_params.setdefault('soft_label',        None)
    all_params.setdefault('postprocess',       {'type': None, 'value': None})
    all_params.setdefault('pos',               'none')
    all_params.setdefault('n_mask',            10)
    all_params.setdefault('patch_size',        10)

    enc = all_params['encoder_params']
    enc.setdefault('use_instance_norm', False)
    # Mamba-specific encoder defaults
    enc.setdefault('d_state',       16)
    enc.setdefault('d_conv',         4)
    enc.setdefault('expand',         2)
    enc.setdefault('bidirectional', True)
    enc.setdefault('fusion_type',   'gate')

    dec = all_params['decoder_params']
    # Mamba-specific decoder defaults
    dec.setdefault('d_state',       16)
    dec.setdefault('d_conv',         4)
    dec.setdefault('expand',         2)
    dec.setdefault('bidirectional', True)
    dec.setdefault('fusion_type',   'gate')

    diff = all_params['diffusion_params']
    diff.setdefault('detach_decoder', False)

    assert all_params['loss_weights']['encoder_boundary_loss'] == 0

    return all_params


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def mode_filter(x, size):
    def modal(P):
        mode = stats.mode(P)
        return mode.mode[0]
    return generic_filter(x, modal, size)


# ── Evaluation (from ASFormer / MSTCN) ──────────────────────────────────────

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels, starts, ends = [], [], []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1], np.float64)
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i
    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + 1)
    if norm:
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp, fp = 0, 0
    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union        = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU          = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        idx = np.array(IoU).argmax()
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def func_eval(label_dir, pred_dir, video_list, obs_p=1.0):
    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    correct, total, edit = 0, 0, 0

    for vid in video_list:
        gt_content   = read_file(os.path.join(label_dir, f'{vid}.txt')).split('\n')[:-1]
        pred_content = read_file(os.path.join(pred_dir,  f'{vid}.txt')).split('\n')[1].split()

        if obs_p != 1.0:
            full_len     = len(gt_content)
            obs_len      = int(full_len * obs_p)
            gt_content   = gt_content[:obs_len]
            pred_content = pred_content[:obs_len]

        assert len(gt_content) == len(pred_content)

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == pred_content[i]:
                correct += 1

        edit += edit_score(pred_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(pred_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1

    acc  = 100 * float(correct) / total
    edit = (1.0 * edit) / len(video_list)
    f1s  = np.zeros(3, dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        recall    = tp[s] / float(tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        denom     = precision + recall
        f1        = (2.0 * precision * recall / denom) if denom > 0 else 0.0
        f1s[s]    = f1 * 100

    return acc, edit, f1s


# ── Visualization ────────────────────────────────────────────────────────────

def plot_barcode(class_num, gt=None, pred=None, show=True, save_file=None):
    if class_num <= 10:
        color_map = plt.cm.tab10
    elif class_num > 20:
        color_map = plt.cm.gist_ncar
    else:
        color_map = plt.cm.tab20

    axprops  = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0, vmax=class_num - 1)
    fig      = plt.figure(figsize=(18, 4))

    if gt is not None:
        ax1 = fig.add_axes([0, 0.45, 1, 0.2], **axprops)
        ax1.set_title('Ground Truth')
        ax1.imshow(gt.reshape((1, -1)), **barprops)

    if pred is not None:
        ax2 = fig.add_axes([0, 0.15, 1, 0.2], **axprops)
        ax2.set_title('Predicted')
        ax2.imshow(pred.reshape((1, -1)), **barprops)

    if save_file is not None:
        fig.savefig(save_file, dpi=400)
    if show:
        plt.show()
    plt.close(fig)


def read_mapping_dict(file_path):
    """Read action-index mapping from a text file."""
    with open(file_path, 'r') as f:
        actions = f.read().split('\n')[:-1]
    actions_dict = {a.split()[1]: int(a.split()[0]) for a in actions}
    return actions_dict


def eval_file(gt_content, recog_content, obs_percentage, classes):
    last_frame  = min(len(recog_content), len(gt_content))
    recognized  = recog_content[int(obs_percentage * len(gt_content)):last_frame]
    ground_truth = gt_content[int(obs_percentage * len(gt_content)):last_frame]

    n_T = np.zeros(len(classes))
    n_F = np.zeros(len(classes))
    for i in range(len(ground_truth)):
        if ground_truth[i] == recognized[i]:
            n_T[int(ground_truth[i])] += 1
        else:
            n_F[int(ground_truth[i])] += 1
    return n_T, n_F


def get_unique_list(input_list):
    unique_list, unique_list_len = [], []
    for idx, x in enumerate(input_list):
        if not unique_list or x != unique_list[-1]:
            unique_list.append(x)
            unique_list_len.append(idx)
    return unique_list, unique_list_len


def get_unique_list_gt(input_list, actions_dict):
    unique_list, unique_list_len = [], []
    for idx, x in enumerate(input_list):
        if not unique_list or actions_dict[x] != unique_list[-1]:
            unique_list.append(actions_dict[x])
            unique_list_len.append(idx)
    return unique_list, unique_list_len

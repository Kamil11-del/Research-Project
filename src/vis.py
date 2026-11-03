import matplotlib.pyplot as plt


# Adapted from ASFormer (https://github.com/ChinaYi/ASFormer/blob/main/eval.py)
def segment_bars(save_path, *labels,
                 gt_unique=None, gt_unique_len=None,
                 pred_unique=None, pred_unique_len=None):
    num_pics  = len(labels)
    color_map = plt.get_cmap('seismic')
    fig, axs  = plt.subplots(num_pics, 1, figsize=(15, num_pics * 1.5))
    barprops  = dict(aspect='auto', cmap=color_map, interpolation='nearest', vmin=0, vmax=50)

    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=1.5)

    for i, label in enumerate(labels):
        ax = axs if num_pics == 1 else axs[i]

        if i == 0:
            name        = 'gt'
            uni_label   = gt_unique
            uni_label_len = gt_unique_len
        else:
            name        = 'actmamba'
            uni_label   = pred_unique
            uni_label_len = pred_unique_len

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{name}', pad=5)
        ax.imshow([label], **barprops)

        img_width = len(label)
        if uni_label is not None and uni_label_len is not None:
            for j, val in enumerate(uni_label):
                if j < len(uni_label_len) - 1:
                    mid_point = (uni_label_len[j] + uni_label_len[j + 1]) / 2
                else:
                    mid_point = (uni_label_len[j] + img_width) / 2
                ax.text(mid_point, 0.6, f'{val}', ha='center', va='top',
                        fontsize=8, color='black', backgroundcolor='white', rotation=330)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    plt.close()

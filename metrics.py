import numpy as np
import torch


def score_extraction(data_loader, model, use_gpu, labelmap=[]):
    with torch.no_grad():
        out_scores, gt_labels = [], []
        for _, data in enumerate(data_loader):
            batch_images, batch_labels = data[0], data[1]
            if use_gpu:
                batch_images = batch_images.cuda()

            if labelmap:
                for i, label in enumerate(labelmap):
                    batch_labels[torch.where(batch_labels==i)] = label

            out_scores.append(model(batch_images))
            gt_labels.extend(batch_labels)

        out_scores = torch.cat(out_scores, 0).data.cpu().numpy()
        gt_labels = np.asarray(gt_labels)

    return out_scores, gt_labels

def mean_top_k_accuracy(scores, labels, k=1):
    idx = np.argsort(-scores, axis=-1)[:, :k]
    labels = np.array(labels)
    matches = np.any(idx == labels.reshape([-1, 1]), axis=-1)

    classes = np.unique(labels)

    accuracy_values = []
    for class_id in classes:
        mask = labels == class_id
        num_valid = np.sum(mask)
        if num_valid == 0:
            continue

        accuracy_values.append(np.sum(matches[mask]) / float(num_valid))

    return np.mean(accuracy_values) if len(accuracy_values) > 0 else 1.0


def mean_average_precision(scores, labels):
    def _ap(in_recall, in_precision):
        mrec = np.concatenate((np.zeros([1, in_recall.shape[1]], dtype=np.float32),
                               in_recall,
                               np.ones([1, in_recall.shape[1]], dtype=np.float32)))
        mpre = np.concatenate((np.zeros([1, in_precision.shape[1]], dtype=np.float32),
                               in_precision,
                               np.zeros([1, in_precision.shape[1]], dtype=np.float32)))

        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        all_ap = []
        cond = mrec[1:] != mrec[:-1]
        for k in range(cond.shape[1]):
            i = np.where(cond[:, k])[0]
            all_ap.append(np.sum((mrec[i + 1, k] - mrec[i, k]) * mpre[i + 1, k]))

        return np.array(all_ap, dtype=np.float32)

    one_hot_labels = np.zeros_like(scores, dtype=np.int32)
    one_hot_labels[np.arange(len(labels)), labels] = 1

    idx = np.argsort(-scores, axis=0)
    sorted_labels = np.take_along_axis(one_hot_labels, idx, axis=0)

    matched = sorted_labels == 1

    tp = np.cumsum(matched, axis=0).astype(np.float32)
    fp = np.cumsum(~matched, axis=0).astype(np.float32)

    num_pos = np.sum(one_hot_labels, axis=0)
    valid_mask = num_pos > 0
    num_pos[~valid_mask] = 1
    num_pos = num_pos.astype(np.float32)

    recall = tp / num_pos.reshape([1, -1])
    precision = tp / (tp + fp)

    ap = _ap(recall, precision)
    valid_ap = ap[valid_mask]
    mean_ap = np.mean(ap) if len(valid_ap) > 0 else 1.0
    return mean_ap

def evaluate_classification(dataloader, model, use_gpu, topk=(1,), labelmap=[]):
    if isinstance(model, torch.nn.Module):
        scores, labels = score_extraction(dataloader, model, use_gpu, labelmap)
    else:
        raise RuntimeError("model isn't Torch one")
    m_ap = mean_average_precision(scores, labels)

    cmc = []
    for k in topk:
        cmc.append(mean_top_k_accuracy(scores, labels, k=k))

    return cmc, m_ap

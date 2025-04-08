import csv
import numpy as np
from sklearn.metrics import confusion_matrix
# By: https://github.com/justchenhao/BIT_CD/tree/adcd7aea6f234586ffffdd4e9959404f96271711
###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        """获得当前混淆矩阵，并计算当前F1得分，并更新混淆矩阵"""
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    harmonic_mean = len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean



def cm2F1(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    return mean_F1

def cm2score(confusion_matrix):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in range(n_class)], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in range(n_class)], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(n_class)], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(n_class)], F1))

    score_dict = {'acc': acc*100, 'miou': mean_iu*100, 'mf1':mean_F1*100}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def cm2score_my(confusion_matrix, class_names):
    hist = confusion_matrix
    n_class = hist.shape[0]
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)
    # ---------------------------------------------------------------------- #
    # 1. Accuracy & Class Accuracy
    # ---------------------------------------------------------------------- #
    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    # acc_cls = np.nanmean(recall)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2*recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)
    # ---------------------------------------------------------------------- #
    # 2. Frequency weighted Accuracy & Mean IoU
    # ---------------------------------------------------------------------- #
    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #
    cls_iou = dict(zip(['iou_'+str(i) for i in class_names], iu))

    cls_precision = dict(zip(['precision_'+str(i) for i in class_names], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in class_names], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in class_names], F1))

    score_dict = {'acc': acc*100, 'miou': mean_iu*100, 'mf1':mean_F1*100}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


def get_confuse_matrix(num_classes, label_gts, label_preds):
    """计算一组预测的混淆矩阵"""
    def __fast_hist(label_gt, label_pred):
        """
        Collect values for Confusion Matrix
        For reference, please see: https://en.wikipedia.org/wiki/Confusion_matrix
        :param label_gt: <np.array> ground-truth
        :param label_pred: <np.array> prediction
        :return: <np.ndarray> values for confusion matrix
        """
        mask = (label_gt >= 0) & (label_gt < num_classes)
        hist = np.bincount(num_classes * label_gt[mask].astype(int) + label_pred[mask],
                           minlength=num_classes**2).reshape(num_classes, num_classes)
        return hist
    confusion_matrix = np.zeros((num_classes, num_classes))
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt.flatten(), lp.flatten())
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']


def calculate_per_class_metrics(preds, labels, class_names=None, num_classes=None):
    """
    Returns a dictionary with per-class accuracy (recall) and IoU,
    along with overall mIoU and mAcc across all classes.
    """
    # class_names = [
    #     "Background clutter",
    #     "Building",
    #     "Road",
    #     "Tree",
    #     "Low vegetation",
    #     "Moving car",
    #     "Static car",
    #     "Human"
    # ]
    if class_names is None and num_classes is None:
        raise ValueError("Either class_names or num_classes must be provided.")
    elif class_names is None:
        class_names = [f"Class_{i}" for i in range(num_classes)]

    num_classes = len(class_names)
    # Flatten preds and labels to 1D numpy arrays
    preds = preds.cpu().numpy().ravel()
    labels = labels.cpu().numpy().ravel()

    # Build confusion matrix (shape: [num_classes, num_classes])
    conf_mat = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    res = cm2score_my(conf_mat, class_names)
    acc = res['acc']
    mIoU = res['miou']
    mF1 = res['mf1']
    iou_classes = [res[f'iou_{class_name}'] for class_name in class_names]
    f1_classes = [res[f'F1_{class_name}'] for class_name in class_names]
    precision_classes = [res[f'precision_{class_name}'] for class_name in class_names]
    recall_classes = [res[f'recall_{class_name}'] for class_name in class_names]
    mAcc = np.average(recall_classes)

    # Prepare a results dictionary
    results = {
        "conf_matrix": conf_mat,
        "per_class_accuracy": recall_classes,
        "per_class_iou": iou_classes,
        "per_class_f1": f1_classes,
        "per_class_precision": precision_classes,
        'accuracy': acc,
        "mean_accuracy": mAcc,
        "mean_iou": mIoU,
        "mean_f1": mF1,
        "class_names": class_names
    }
    return results


def print_class_metrics_table(metrics_dict):
    """
    Nicely prints a table of per-class accuracy & IoU
    (similar to the UAVID style).
    """
    class_names = metrics_dict["class_names"]
    accs = metrics_dict["per_class_accuracy"]
    ious = metrics_dict["per_class_iou"]
    mAcc = metrics_dict["mean_accuracy"]
    mIoU = metrics_dict["mean_iou"]

    print("\nPer-class results:")
    print(f"{'Class':<22}  {'Acc (%)':>8}  {'IoU (%)':>8}")
    print("-" * 42)
    for i, cname in enumerate(class_names):
        print(f"{cname:<22}  {accs[i] * 100:8.2f}  {ious[i] * 100:8.2f}")
    print("-" * 42)
    print(f"{'Mean':<22}  {mAcc * 100:8.2f}  {mIoU * 1:8.2f}")


def write_epoch_csv(epoch_data, csv_file):
    """
    Writes a CSV with this structure:

    Row 0: "", "Epoch1", "Epoch2", ...
    Row 1..: each row is a metric:
        acc
        mIoU
        mF1
        mAcc
        iou_{class_name0}
        F1_{class_name0}
        recall_{class_name0}
        precision_{class_name0}
        iou_{class_name1}
        ...
    And each row has one cell per epoch's value.

    Args:
      epoch_data: dict of { epoch_number: results },
                  where 'results' is the dictionary from calculate_per_class_metrics(...).
      csv_file:   path to the CSV file to write.
    """

    # 1) Sort epoch keys so columns appear in ascending order
    sorted_epochs = sorted(epoch_data.keys())
    # Each 'results' in epoch_data[ep] is the dictionary returned by your calculate_per_class_metrics.

    # 2) First CSV row: blank in col A, then "Epoch1", "Epoch2", ...
    header_row = [""]
    for ep in sorted_epochs:
        header_row.append(f"Epoch{ep}")

    # We’ll accumulate rows in a list of lists, then write them at the end
    rows = [header_row]

    # 3) A helper to generate a row for one metric. This will read from each epoch's dict
    def build_metric_row(metric_name, fetch_value_fn):
        """
        metric_name: string for column A
        fetch_value_fn: function(results_dict) -> float
                        extracts the float metric from the dictionary
        """
        row = [metric_name]
        for ep in sorted_epochs:
            results = epoch_data[ep]
            val = fetch_value_fn(results)
            row.append(f"{val:.4f}")
        return row

    # 4) Overall metrics
    #    from your dictionary:
    #    'accuracy', 'mean_iou', 'mean_f1', 'mean_accuracy'
    #    each is a single float
    rows.append(build_metric_row("acc", lambda r: r["accuracy"]))  # overall ACC
    rows.append(build_metric_row("mIoU", lambda r: r["mean_iou"]))  # overall IoU
    rows.append(build_metric_row("mF1", lambda r: r["mean_f1"]))  # overall F1
    rows.append(build_metric_row("mAcc", lambda r: r["mean_accuracy"]))  # overall mAcc

    # 5) Per-class metrics
    #    "per_class_iou" => array of length num_classes
    #    "per_class_f1"
    #    "per_class_accuracy" => these are recall
    #    "per_class_precision"
    #    plus "class_names"
    # We'll assume all epochs have the same class_names:
    first_epoch = sorted_epochs[0]
    class_names = epoch_data[first_epoch]["class_names"]
    num_classes = len(class_names)

    for i, cname in enumerate(class_names):
        # iou_{class_name}
        rows.append(build_metric_row(f"iou_{cname}",
                                     lambda r: r["per_class_iou"][i]))
        # F1_{class_name}
        rows.append(build_metric_row(f"F1_{cname}",
                                     lambda r: r["per_class_f1"][i]))
        # recall_{class_name}
        rows.append(build_metric_row(f"recall_{cname}",
                                     lambda r: r["per_class_accuracy"][i]))  # stored as 'per_class_accuracy'
        # precision_{class_name}
        rows.append(build_metric_row(f"precision_{cname}",
                                     lambda r: r["per_class_precision"][i]))

    # 6) Write all rows to the CSV
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)

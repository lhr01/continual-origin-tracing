import os
import numpy as np
import torch


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=2):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around(
        (y_pred == y_true).sum() * 100 / len(y_true), decimals=2
    )

    # Grouped accuracy
    for class_id in range(0, np.max(y_true) + 1, increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true < class_id + increment)
        )[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"), str(class_id + increment - 1).rjust(2, "0")
        )
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]

    all_acc["old"] = (
        0
        if len(idxes) == 0
        else np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
        )
    )

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2
    )

    return all_acc

def show_detail_acc(_accy, data_manager):
    # 创建一个新的字典来存储结果
    _class_accuracy = {}
    # 遍历字典 b 的键值对
    for k, v in _accy["pre_class_accuracy"].items():
        # 使用字典 a 中的值替换字典 b 中的键
        class_name = data_manager._idx_to_class.get(k)
        _class_accuracy[class_name] = v

    # 创建一个正确和错误的字典来存储正确和错误的结果
    _misclassifications = {}
    for k, v in _accy["misclassifications"].items():
        per_misclassifications = {}
        for per_k, per_v in v.items():
            per_class_name = data_manager._idx_to_class.get(per_k)
            per_misclassifications[per_class_name] = per_v
        class_name = data_manager._idx_to_class.get(k)
        _misclassifications[class_name] = per_misclassifications

    return _class_accuracy, _misclassifications
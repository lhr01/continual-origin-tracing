import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters, show_detail_acc
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}".format(args["model_name"],args["llm_type"],args["dataset"])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}".format(
        args["model_name"],
        args["llm_type"],
        args["dataset"],
        args["M"],
    )

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["llm_type"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
    )
    model = factory.get_model(args["model_name"], args)

    stage_curve = {"top1": []}
    for stage in range(data_manager.nb_stages):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)
        stage_accy = model.eval_task()
        model.after_task()

        logging.info("Stage: {}".format(stage_accy["grouped"]))

        stage_curve["top1"].append(stage_accy["top1"])
        # 创建一个新的字典只存储正确的结果
        class_accuracy = {}

        # 遍历字典 b 的键值对
        for k, v in stage_accy["pre_class_accuracy"].items():
            # 使用字典 a 中的值替换字典 b 中的键
            class_name = data_manager._idx_to_class.get(k)
            class_accuracy[class_name] = v

        logging.info("Accuracy per class: {}".format(class_accuracy))
        # 创建一个正确和错误的字典来存储正确和错误的结果
        misclassifications = {}
        for k, v in stage_accy["misclassifications"].items():
            per_misclassifications = {}
            for per_k, per_v in v.items():
                per_class_name = data_manager._idx_to_class.get(per_k)
                per_misclassifications[per_class_name] = per_v
            class_name = data_manager._idx_to_class.get(k)
            misclassifications[class_name] = per_misclassifications

        logging.info("Stage Acc and Err per class: {}".format(misclassifications))
        logging.info("top1 curve: {}".format(stage_curve["top1"]))

        print('Average Accuracy:', sum(stage_curve["top1"])/len(stage_curve["top1"]))
        logging.info("Average Accuracy: {}".format(sum(stage_curve["top1"])/len(stage_curve["top1"])))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))

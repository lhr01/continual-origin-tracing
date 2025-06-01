import logging
import numpy as np
import torch
from transformers import AutoTokenizer
from torchvision import transforms
from utils.data import COT_Bench, textsDataset, LlamaDataset

class DataManager(object):
    def __init__(self, dataset_name, llm_type, shuffle, seed, init_cls, increment):
        self.dataset_name = dataset_name
        self.llm_type = llm_type
        self._setup_data_text(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_stages(self):
        return len(self._increments)

    def get_stage_size(self, stage):
        return self._increments[stage]

    def get_total_classnum(self):
        return len(self._class_order)
    
    def get_texts_dataset(
        self, indices, source, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "dev":
            x, y = self._dev_data, self._dev_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        tokenizer  = _get_tokenizer(self.llm_type)
        text_Dataset = None
        if 'roberta' in self.llm_type:
            text_Dataset = textsDataset(tokenizer, data, targets)
        elif 'Llama' in self.llm_type:
            text_Dataset = LlamaDataset(tokenizer, data, targets)

        if ret_data:
            return data, targets, text_Dataset
        else:
            return text_Dataset

    def _setup_data_text(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data(self.dataset_name)
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._dev_data, self._dev_targets = idata.dev_data, idata.dev_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._class_to_idx, self._classes = idata.class_to_idx, idata.classes
        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        # class to index change
        for key, value in self._class_to_idx.items():
            self._class_to_idx[key] = self._class_order.index(value)

        self._idx_to_class = {v: k for k, v in self._class_to_idx.items()}
        
        # Map indices
        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._dev_targets = _map_new_class_index(self._dev_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)


    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if "cot-bench" in name:
        return COT_Bench()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

def _get_tokenizer(llm_type):

    tokenizer = None

    if llm_type == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("../../model/roberta-base")
    elif llm_type == 'Llama-2-7b':
        tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-2-7b")
    elif llm_type == 'Llama-3.2-1B':
        tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-3.2-1B")
    elif llm_type == 'Llama-3.2-1B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-3.2-1B-Instruct")
    elif llm_type == 'Llama-3-8B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained("../../model/Llama-3-8B-Instruct")
    elif llm_type == 'Qwen2.5-7B-Instruct':
        tokenizer = AutoTokenizer.from_pretrained("../../model/Qwen2.5-7B-Instruct")

    return tokenizer
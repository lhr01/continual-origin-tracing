import logging
import numpy as np
import torch
import copy
from torch import nn
from torch.serialization import load
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import SimpleLLMNet
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleLLMNet(args, True)
        self.args=args

    def after_task(self):
        self._known_classes = self._total_classes
    
    def replace_fc(self,trainloader, dev_loader, model, args):
        model = model.eval()

        if self.args['use_RP']:
            #these lines are needed because the CosineLinear head gets deleted between streams and replaced by one with more classes (for CIL)
            self._network.multi_local_fcs.use_RP=True
            self._network.global_fc.use_RP=True
            if self.args['M']>0:
                self._network.multi_local_fcs.W_rand=self.W_rand
                self._network.global_fc.W_rand=self.W_rand
            else:
                self._network.multi_local_fcs.W_rand=None
                self._network.global_fc.W_rand=None

        train_Features_f = []
        train_label_list = []
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                batch = tuple(t.cuda() for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }
                embedding=model.llm(**inputs)
                train_Features_f.append(embedding.cpu())
                train_label_list.append(inputs['labels'].cpu())
        train_Features_f = torch.cat(train_Features_f, dim=0)
        train_label_list = torch.cat(train_label_list, dim=0)

        train_Y=target2onehot(train_label_list,self.total_classnum)

        dev_Features_f = []
        dev_label_list = []
        with torch.no_grad():
            for i, batch in enumerate(dev_loader):
                batch = tuple(t.cuda() for t in batch)
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "labels": batch[2]
                }
                embedding=model.llm(**inputs)
                dev_Features_f.append(embedding.cpu())
                dev_label_list.append(inputs['labels'].cpu())
        dev_Features_f = torch.cat(dev_Features_f, dim=0)
        dev_label_list = torch.cat(dev_label_list, dim=0)

        dev_Y=target2onehot(dev_label_list,self.total_classnum)
        if self.args['use_RP']:
            if self.args['M']>0:
                train_Features_h=torch.nn.functional.relu(train_Features_f@ self._network.global_fc.W_rand.cpu())
                dev_Features_h=torch.nn.functional.relu(dev_Features_f@ self._network.global_fc.W_rand.cpu())
            else:
                train_Features_h=train_Features_f
                dev_Features_h=dev_Features_f

            unique_labels = list(set(train_label_list.tolist()))
            for cls in unique_labels:
                cls_indices = (train_label_list == cls)
                Features_h_cls = train_Features_h[cls_indices]
                Y_cls = train_Y[cls_indices]
                self.per_Q[cls] = Features_h_cls.T @ Y_cls
                self.per_G[cls] = Features_h_cls.T @ Features_h_cls
            
            self.cal_multi_local_fcs_wo(train_Features_h, train_Y, dev_Features_h, dev_Y)
            self.cal_global_fc_wo(train_Features_h, train_Y, dev_Features_h, dev_Y)
        else:
            for class_index in np.unique(self.train_dataset.labels):
                data_index=(train_label_list==class_index).nonzero().squeeze(-1)
                if self.is_dil:
                    class_prototype=train_Features_f[data_index].sum(0)
                    self._network.global_fc.weight.data[class_index]+=class_prototype.to(device='cuda')
                else:
                    class_prototype=train_Features_f[data_index].mean(0)
                    self._network.global_fc.weight.data[class_index]=class_prototype

    def optimise_ridge_parameter(self,train_Features, train_Y, dev_Features, dev_Y):
        ridges=10.0**np.arange(-1,6)
        losses=[]
        Q_val = train_Features.T @ train_Y 
        G_val = train_Features.T @ train_Features
        for ridge in ridges:
            Wo=torch.linalg.solve(G_val+ridge*torch.eye(G_val.size(dim=0)),Q_val).T
            Y_dev_pred=dev_Features@Wo.T
            losses.append(F.mse_loss(Y_dev_pred,dev_Y))
        ridge=ridges[np.argmin(np.array(losses))]
        # logging.info("Optimal lambda: "+str(ridge))
        return ridge

    def incremental_train(self, data_manager):
        self.total_classnum = data_manager.get_total_classnum()
        self._cur_stage += 1
        self._total_classes = self._known_classes + data_manager.get_stage_size(self._cur_stage)
        if self.args['use_RP']:
            #temporarily remove RP weights
            del self._network.global_fc
            self._network.global_fc=None
        if self.args['use_RP'] and self.args['M'] > 0:
            self._network.updata_multi_local_fcs(self._cur_stage, 2, use_RP=self.args['use_RP'], M=self.args['M'], pre_classes=self._known_classes, total_classes=self._total_classes)
            self._network.update_global_fc(self._total_classes, use_RP=self.args['use_RP'], M=self.args['M'] )
        else:
            self._network.updata_multi_local_fcs(self._cur_stage, 2, pre_classes=self._known_classes, total_classes=self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))
        self.data_manager=data_manager

        train_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="train",)
        self.train_dataset = train_dataset
        self.train_loader = DataLoader(train_dataset, batch_size=self.args['batch_size'], shuffle=True)

        dev_dataset = data_manager.get_texts_dataset(np.arange(self._known_classes, self._total_classes), source="dev",)
        self.dev_loader = DataLoader(dev_dataset, batch_size=self.args['batch_size'], shuffle=True)
        
        test_dataset = data_manager.get_texts_dataset(np.arange(0, self._total_classes), source="test",)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args['batch_size'], shuffle=False)

        self._train(self.train_loader, self.test_loader, self.dev_loader)


    def _train(self, train_loader, test_loader, dev_loader):
        
        self._network.to(self._device)
        if self._cur_stage == 0 and self.args['use_RP']:
            self.setup_RP()
        self.replace_fc(train_loader, dev_loader, self._network, None)
        
    def setup_RP(self):
        self.initiated_G=False
        self._network.multi_local_fcs.use_RP=True
        self._network.global_fc.use_RP=True
        if self.args['M']>0:
            #RP with M > 0
            M=self.args['M']
            self._network.multi_local_fcs.W_rand=torch.randn(self._network.multi_local_fcs.in_features,M).to(device='cuda')
            self.W_rand=copy.deepcopy(self._network.multi_local_fcs.W_rand) #make a copy that gets passed each time the head is replaced
            self._network.global_fc.weight = nn.Parameter(torch.Tensor(self._network.global_fc.out_features, M).to(device='cuda')) #num classes in task x M
            self._network.global_fc.reset_parameters()
            self._network.global_fc.W_rand=copy.deepcopy(self._network.multi_local_fcs.W_rand)
        else:
            #no RP, only decorrelation
            M=self._network.multi_local_fcs.in_features #this M is L in the paper
        self.per_Q=torch.zeros(self.total_classnum, M, self.total_classnum)
        self.per_G=torch.zeros(self.total_classnum, M, M)
        self.Q=torch.zeros(M,self.total_classnum)
        self.G=torch.zeros(M,M)

    def cal_multi_local_fcs_wo(self, train_Features_h, train_Y, dev_Features_h, dev_Y):
        total_class_num = self._total_classes
        before_class_num = self._known_classes
        for cur_index in range(before_class_num, total_class_num):
            for before_index in range(0, cur_index):
                combine_Q = self.per_Q[before_index] + self.per_Q[cur_index]
                combine_G = self.per_G[before_index] + self.per_G[cur_index]
                ridge=self.optimise_ridge_parameter(train_Features_h, train_Y, dev_Features_h, dev_Y)
                logging.info(f"local {before_index}-{cur_index}," + " Optimal lambda: " + str(ridge))
                Wo=torch.linalg.solve(combine_G+ridge*torch.eye(combine_G.size(dim=0)),combine_Q).T
                key = f"{before_index}-{cur_index}"
                self._network.multi_local_fcs.local_heads[key].weight.data=Wo[[before_index,cur_index],:].to(device='cuda')


    def cal_global_fc_wo(self, train_Features_h, train_Y, dev_Features_h, dev_Y):
        self.Q = self.Q + train_Features_h.T @ train_Y 
        self.G = self.G + train_Features_h.T @ train_Features_h
        ridge = self.optimise_ridge_parameter(train_Features_h, train_Y, dev_Features_h, dev_Y)
        logging.info(f"global," + " Optimal lambda: " + str(ridge))
        Wo = torch.linalg.solve(self.G+ridge * torch.eye(self.G.size(dim=0)), self.Q).T
        self._network.global_fc.weight.data=Wo[0 : self._network.global_fc.weight.shape[0],:].to(device='cuda')
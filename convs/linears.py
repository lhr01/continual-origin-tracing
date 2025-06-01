'''
Reference:
https://github.com/hshustc/CVPR19_Incremental_Learning/blob/master/cifar100-class-incremental/modified_linear.py
'''
import math
import torch
from torch import nn
from torch.nn import functional as F


class SimpleLinear(nn.Module):
    '''
    Reference:
    https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
    '''
    def __init__(self, in_features, out_features, bias=True):
        super(SimpleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='linear')
        nn.init.constant_(self.bias, 0)

    def forward(self, input):
        return {'logits': F.linear(input, self.weight, self.bias)}


class CosineLinear(nn.Module):
    def __init__(self, in_features, out_features, use_RP=False, M=0, nb_proxy=1, to_reduce=False, sigma=True, parent=None):
        super(CosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features * nb_proxy
        self.nb_proxy = nb_proxy
        self.to_reduce = to_reduce
        self.use_RP=use_RP
        self.M=M
        self.parent = parent
        if use_RP and M > 0:
            self.weight = nn.Parameter(torch.Tensor(self.out_features, M))
        else:
            self.weight = nn.Parameter(torch.Tensor(self.out_features, in_features))
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input, W_rand=None):
        if not self.use_RP:
            out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        else:
            if W_rand is not None:
                inn = torch.nn.functional.relu(input @ W_rand)
            elif self.W_rand is not None:
                inn = torch.nn.functional.relu(input @ self.W_rand)
            else:
                inn=input
                #inn=torch.bmm(input[:,0:100].unsqueeze(-1), input[:,0:100].unsqueeze(-2)).flatten(start_dim=1) #interaction terms instead of RP
            out = F.linear(inn,self.weight)
            out = F.softmax(out, dim=1)
        if self.to_reduce:
            # Reduce_proxy
            out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {'logits': out}



class MultiLocalCosineLinear(nn.Module):
    def __init__(self, in_features, out_features, use_RP=False, M=0):
        super(MultiLocalCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_RP = use_RP
        self.M = M
        self.local_heads = nn.ModuleDict()

    def add_local_fc(self, in_dim, out_dim, pre_classes, total_classes):
        for i in range(pre_classes, total_classes):
            for j in range(0, i):
                key = f"{j}-{i}"
                self.local_heads[key] = CosineLinear(in_dim, out_dim, self.use_RP, self.M)

    def local_detect(self, x, first_out, cur_stage):
        _, top2_indices = torch.topk(first_out, k=2, dim=1)
        sorted_indices, _ = torch.sort(top2_indices, dim=1)

        preds_list = []
        for i in range(sorted_indices.size(0)):
            first_kind = sorted_indices[i][0].tolist()
            second_kind = sorted_indices[i][1].tolist()
            single_sample = x[i].unsqueeze(0)
            key = f"{first_kind}-{second_kind}"
            two_head_logits = self.local_heads[key](single_sample, W_rand=self.W_rand)['logits']
            select_kind = torch.topk(two_head_logits, k=1, dim=1)[1].view(-1)
            preds_list.append(sorted_indices[i][select_kind].tolist())

        return preds_list
        
    def cal_local_pred(self, single_sample_logits, cur_stage):
        cur_index = cur_stage + 1
        for stage_index in range(cur_index):

            if stage_index+1 == cur_index: 
                return stage_index if single_sample_logits[stage_index][0][0] > single_sample_logits[stage_index][0][1] else cur_stage + 1
            
            output_list = [0] * (len(single_sample_logits[stage_index]) + 1)
            for head_index, (first, second) in enumerate(single_sample_logits[stage_index]):
                output_list[0] += first
                output_list[head_index + 1] = second
            output_list[0] /= len(single_sample_logits[stage_index])
            is_largest = all(output_list[0] > output_list[i] for i in range(1, len(output_list)))
            if is_largest:
                return stage_index

    def all_detect(self, x, cur_stage):
        cur_index = cur_stage + 1
        logits_list = []
        for i in range(x.size(0)):    # x.size(0) == batch size
            single_sample = x[i].unsqueeze(0) 
            single_sample_logits = []
            for before_index in range(cur_index):  
                task_logits = []
                for head_index in range(cur_index - before_index):
                    task_head_logits = self.tasks_heads[before_index][head_index](single_sample,W_rand=self.W_rand)['logits']
                    task_logits.append(task_head_logits.squeeze(0))
                single_sample_logits.append(task_logits)
            logits = self.cal_all_pred(single_sample_logits, cur_stage)
            logits_list.append(logits)

        return logits_list

    def cal_all_pred(self,  single_sample_logits, cur_stage):
        cur_index = cur_stage + 1
        output_list = [0] * (len(single_sample_logits[0]) + 1)
        for task_index in range(cur_index):
            for head_index, (first, second) in enumerate(single_sample_logits[task_index]):
                output_list[task_index] += first
                output_list[task_index + head_index + 1] += second
        return output_list

    def forward(self, x, first_out, cur_stage, is_all=False):

        if is_all:
            logits = self.all_detect(x, cur_stage)
        else:
            logits = self.local_detect(x, first_out, cur_stage)

        return {'logits': logits}

class SplitCosineLinear(nn.Module):
    def __init__(self, in_features, out_features1, out_features2, nb_proxy=1, sigma=True):
        super(SplitCosineLinear, self).__init__()
        self.in_features = in_features
        self.out_features = (out_features1 + out_features2) * nb_proxy
        self.nb_proxy = nb_proxy
        self.fc1 = CosineLinear(in_features, out_features1, nb_proxy, False, False)
        self.fc2 = CosineLinear(in_features, out_features2, nb_proxy, False, False)
        if sigma:
            self.sigma = nn.Parameter(torch.Tensor(1))
            self.sigma.data.fill_(1)
        else:
            self.register_parameter('sigma', None)

    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.fc2(x)

        out = torch.cat((out1['logits'], out2['logits']), dim=1)  # concatenate along the channel

        # Reduce_proxy
        out = reduce_proxies(out, self.nb_proxy)

        if self.sigma is not None:
            out = self.sigma * out

        return {
            'old_scores': reduce_proxies(out1['logits'], self.nb_proxy),
            'new_scores': reduce_proxies(out2['logits'], self.nb_proxy),
            'logits': out
        }


def reduce_proxies(out, nb_proxy):
    if nb_proxy == 1:
        return out
    bs = out.shape[0]
    nb_classes = out.shape[1] / nb_proxy
    assert nb_classes.is_integer(), 'Shape error'
    nb_classes = int(nb_classes)

    simi_per_class = out.view(bs, nb_classes, nb_proxy)
    attentions = F.softmax(simi_per_class, dim=-1)

    return (attentions * simi_per_class).sum(-1)


import copy
import torch
from torch import nn
from transformers import AutoConfig, AutoTokenizer, RobertaModel
from convs.linears import SimpleLinear, CosineLinear, MultiLocalCosineLinear
from convs.simple_roberta import RobertaForSimple
from convs.simple_llama import LlamaForSimple
from convs.simple_qwen2 import Qwen2Forsimple
def get_llm_network(args, pretrained=False):
    name = args["llm_type"].lower()
    if name=='roberta-base':
        config = AutoConfig.from_pretrained("../../model/roberta-base")
        model = RobertaForSimple(config)
        checkpoint_model=RobertaModel.from_pretrained("../../model/roberta-base")
        state_dict = checkpoint_model.state_dict()
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
        model.out_dim=768
        return model.eval()
    elif name=="llama-2-7b":
        config = AutoConfig.from_pretrained("../../model/Llama-2-7b",low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = LlamaForSimple.from_pretrained("../../model/Llama-2-7b", config=config)
        model.out_dim=4096
        return model.eval()
    elif name=="llama-3.2-1b":
        config = AutoConfig.from_pretrained("../../model/Llama-3.2-1B",low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = LlamaForSimple.from_pretrained("../../model/Llama-3.2-1B", config=config)
        model.out_dim=2048
        return model.eval()
    elif name=="llama-3.2-1b-instruct":
        config = AutoConfig.from_pretrained("../../model/Llama-3.2-1B-Instruct",low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = LlamaForSimple.from_pretrained("../../model/Llama-3.2-1B-Instruct", config=config)
        model.out_dim=2048
        return model.eval()
    elif name=="llama-3-8b-instruct":
        config = AutoConfig.from_pretrained("../../model/Llama-3-8B-Instruct",low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = LlamaForSimple.from_pretrained("../../model/Llama-3-8B-Instruct", config=config)
        model.out_dim=4096
        return model.eval()
    elif name=="qwen2.5-7b-instruct":
        config = AutoConfig.from_pretrained("../../model/Qwen2.5-7B-Instruct",low_cpu_mem_usage=True, torch_dtype=torch.float16)
        model = Qwen2Forsimple.from_pretrained("../../model/Qwen2.5-7B-Instruct", config=config)
        model.out_dim=3584
        return model.eval()
    else:
        raise NotImplementedError("Unknown type {}".format(name))

class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        print('This is for the BaseNet initialization.')
        self.llm = get_llm_network(args, pretrained)
        print('After BaseNet initialization.')
        self.multi_local_fcs = None
        self.global_fc = None

    @property
    def feature_dim(self):
        return self.llm.out_dim

    def extract_vector(self, x):
        return self.llm(x)["features"]

    def forward(self, task_ids, head_ids, inputs):
        x = self.llm(**inputs)
        out = self.global_fc(task_ids, head_ids, x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def update_fc(self, nb_classes):
        pass

    def generate_fc(self, in_dim, out_dim):
        pass

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self


class SimpleLLMNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)

    def update_global_fc(self, nb_classes, nextperiod_initialization=None, use_RP=False, M=0):
        global_fc = self.generate_fc(self.feature_dim, nb_classes, use_RP=use_RP, M=M).cuda()
        if self.global_fc is not None:
            nb_output = self.global_fc.out_features
            weight = copy.deepcopy(self.global_fc.weight.data)
            global_fc.sigma.data = self.global_fc.sigma.data
            if nextperiod_initialization is not None:
                weight = torch.cat([weight, nextperiod_initialization])
            else:
                weight = torch.cat([weight, torch.zeros(nb_classes - nb_output, self.feature_dim).cuda()])
            global_fc.weight = nn.Parameter(weight)
        del self.global_fc
        self.global_fc = global_fc

    def updata_multi_local_fcs(self, cur_task, nb_classes, use_RP=False, M=0, pre_classes=0, total_classes=0):
        if cur_task == 0:
            self.multi_local_fcs = self.generate_multi_local_fcs(self.feature_dim, nb_classes, use_RP=use_RP, M=M).cuda()
        self.multi_local_fcs.add_local_fc(self.feature_dim, nb_classes, pre_classes, total_classes)

    def generate_multi_local_fcs(self, in_dim, out_dim, use_RP=False, M=0):
        multi_local_fcs = MultiLocalCosineLinear(in_dim, out_dim, use_RP=use_RP, M=M)
        return multi_local_fcs

    def generate_fc(self, in_dim, out_dim, use_RP=False, M=0):
        fc = CosineLinear(in_dim, out_dim, use_RP=use_RP, M=M)
        return fc

    def extract_vector(self, inputs):
        return self.llm(**inputs)

    def forward(self, inputs, cur_stage, is_all=False):
        x = self.llm(**inputs)
        first_out = self.global_fc(x)['logits']
        second_out = self.multi_local_fcs(x, first_out, cur_stage, is_all)
        # out.update(x)
        return first_out, second_out

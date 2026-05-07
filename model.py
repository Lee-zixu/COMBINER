import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch.distributions import Categorical
from torch.autograd import Variable
from transformers import CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer, CLIPTextModelWithProjection
import open_clip
import os
from typing import Callable, Optional, Sequence, Tuple


class SAA(nn.Module):
    def __init__(self, num_units, attention_unit_size, num_classes, type='local'):
        super(SAA, self).__init__()
        attention_unit_size = int(num_units / 2)
        self.fc1 = nn.Linear(num_units, attention_unit_size, bias=False)  
        self.fc2 = nn.Linear(attention_unit_size, num_classes, bias=False) 
        if type == 'global':
            self.softDim = -1
        else:
            self.softDim = -1

    def forward(self, input_x, input_y=None): 
        attention_matrix = self.fc2(torch.tanh(self.fc1(input_x))).transpose(1, 2) 
        attention_weight = torch.softmax(attention_matrix, dim=self.softDim)
        attention_out = torch.matmul(attention_weight, input_x) 
        return attention_out

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class Backbone(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8):
        super().__init__()
        self.clip, _, _ = open_clip.create_model_and_transforms('ViT-H-14', pretrained=os.path.join('./Pretrain_Model/CLIP-ViT-H-14-laion2B-s32B-b79K', 'open_clip_pytorch_model.bin'))
        self.clip = self.clip.float()
        self.tokenizer = open_clip.get_tokenizer('ViT-H-14')
        self.hidden_dim = hidden_dim
        self.fc = nn.Linear(1280,1024)
        self.text_fc = nn.Linear(1024,1024)

        self.local_SAA = SAA(hidden_dim, hidden_dim, local_token_num)
        self.global_SAA = SAA(hidden_dim, hidden_dim, global_token_num)

        self.local_token_num = local_token_num
        self.global_token_num = global_token_num
        

    def visual_out(self, x):
        x = self.clip.visual.conv1(x)  
        x = x.reshape(x.shape[0], x.shape[1], -1)  
        x = x.permute(0, 2, 1) 

        x = torch.cat([_expand_token(self.clip.visual.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        x = x + self.clip.visual.positional_embedding.to(x.dtype)

        x = self.clip.visual.patch_dropout(x)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2) 
        x = self.clip.visual.transformer(x)
        x = x.permute(1, 0, 2) 

        x = self.clip.visual.ln_post(x)
        pooled, tokens = self.clip.visual._global_pool(x)

        pooled = pooled @ self.clip.visual.proj

        
        return pooled, x

    def text_out(self, text):
        cast_dtype = self.clip.transformer.get_cast_dtype()

        x = self.clip.token_embedding(text).to(cast_dtype)  

        x = x + self.clip.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2) 
        x = self.clip.transformer(x, attn_mask=self.clip.attn_mask)
        x = x.permute(1, 0, 2) 
        x = self.clip.ln_final(x) 
        pooled, tokens = text_global_pool(x, text, self.clip.text_pool_type)
        if self.clip.text_projection is not None:
            if isinstance(self.clip.text_projection, nn.Linear):
                pooled = self.clip.text_projection(x)
            else:
                pooled = pooled @ self.clip.text_projection

        return pooled, x


    def extract_img_fea(self, x):
        global_fea, x = self.visual_out(x)
        global_attribute_prototype_features = self.global_SAA(global_fea.unsqueeze(1))
        local_attribute_prototype_features = self.local_SAA(self.fc(x.float()))
        return torch.cat([global_attribute_prototype_features, local_attribute_prototype_features], dim=1), (global_fea.unsqueeze(1).repeat(1, self.global_token_num, 1), x)
    
    def extract_text_fea(self, txt):
        txt = self.tokenizer(txt).cuda()
        global_fea, x = self.text_out(txt)
        global_attribute_prototype_features = self.global_SAA(global_fea.unsqueeze(1))
        local_attribute_prototype_features = self.local_SAA(self.text_fc(x.float()))

        return torch.cat([global_attribute_prototype_features, local_attribute_prototype_features], dim=1), (global_fea.unsqueeze(1).repeat(1, self.global_token_num, 1), x)

class COMBINER(nn.Module):
    def __init__(self, hidden_dim=1024, dropout=0.0, local_token_num=8, global_token_num=8, t=10):
        super().__init__()
        self.backbone = Backbone(hidden_dim, dropout, local_token_num, global_token_num)
        self.loss_T = nn.Parameter(torch.tensor([10.]))
        self.local_weight = nn.Parameter(torch.tensor([1.0 for _ in range(local_token_num + global_token_num)]))

        self.remain_map = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.t = t
        self.compose_SAA =  nn.Sequential(
            SAA(hidden_dim, hidden_dim, (global_token_num+local_token_num) * 2),
            nn.Tanh(),
            SAA(hidden_dim, hidden_dim, (global_token_num+local_token_num)),
        )

    def target_fea(self, tag):
        tar_multi_grained_att_proto_fea, (_, _) = self.backbone.extract_img_fea(tag)
        fuse_local = self.compose_SAA(torch.cat([tar_multi_grained_att_proto_fea, tar_multi_grained_att_proto_fea], dim=1))
        ref_mask = self.remain_map(torch.cat([fuse_local, tar_multi_grained_att_proto_fea], dim=-1))
        tar_multi_grained_att_proto_fea = ref_mask * tar_multi_grained_att_proto_fea
        return tar_multi_grained_att_proto_fea, fuse_local, ref_mask
    
    def compose_feature(self, ref, mod):
        ref_multi_grained_att_proto_fea, (_, _) = self.backbone.extract_img_fea(ref)
        mod_multi_grained_att_proto_fea, (_, _) = self.backbone.extract_text_fea(mod)
        
        CUP = self.compose_SAA(torch.cat([ref_multi_grained_att_proto_fea, mod_multi_grained_att_proto_fea], dim=1))
        ref_sp_remap = self.remain_map(torch.cat([CUP, ref_multi_grained_att_proto_fea], dim=-1))
        mod_sp_remap = self.remain_map(torch.cat([CUP, mod_multi_grained_att_proto_fea], dim=-1)) 
        fuse_local = ref_sp_remap * ref_multi_grained_att_proto_fea + mod_sp_remap * mod_multi_grained_att_proto_fea

        return fuse_local, CUP, ref_sp_remap, mod_sp_remap, ref_multi_grained_att_proto_fea, mod_multi_grained_att_proto_fea

    def extract_retrieval_compose(self, ref, mod):
        fuse_local, _, _, _, _, _= self.compose_feature(ref, mod)
        fuse_local = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)

        return fuse_local

    def extract_retrieval_target(self, tag):
        tag_local, fuse_local, ref_mask = self.target_fea(tag)
        tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
        return tag_local

    def compute_loss(self, ref, mod, tag, cluster_result=None, index=None):
        fuse_local, CUP, ref_mask, mod_mask, ref_multi_grained_att_proto_fea, mod_multi_grained_att_proto_fea = self.compose_feature(ref, mod)
        tag_local, tag_fuse, tag_mask = self.target_fea(tag)
        loss = {}

        retrieval_query = F.normalize(torch.mean(fuse_local, dim=1), p=2, dim=-1)
        retrieval_target = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)

        tag_feature = (F.normalize(tag_local, p=2, dim=-1) * self.local_weight.unsqueeze(0).unsqueeze(-1)).flatten(1)
        loss['rank'] = self.info_nce(retrieval_query, retrieval_target)
        loss['kl_p'] = self.kl_div(retrieval_query, retrieval_target, tag_feature, tag_feature, self.t) 
        # loss['mask'] = F.mse_loss(ref_mask, tag_mask) + F.mse_loss(mod_mask, tag_mask)

        loss['cls'] = torch.tensor(0., device=fuse_local.device)
        loss['kl_c'] = torch.tensor(0., device=fuse_local.device)
        proto_loss = torch.tensor(0., device=fuse_local.device)
        kl_loss = torch.tensor(0., device=fuse_local.device)
        if cluster_result is not None:
            mod_img1 = retrieval_query
            img2 = retrieval_target
            for n, (im2cluster, prototypes, density) in enumerate(zip(cluster_result['im2cluster'], cluster_result['centroids'], cluster_result['density'])):
                pos_proto_id = im2cluster[index:(index + tag.shape[0])]
                pos_prototypes = prototypes[pos_proto_id]
                proto_selected = pos_prototypes
                logits_proto_modimg = torch.mm(mod_img1, proto_selected.t())
                logits_proto_img2 = torch.mm(img2, proto_selected.t())

                labels_proto = torch.linspace(0, mod_img1.size(0) - 1, steps=mod_img1.size(0)).long().cuda()
                proto_loss = proto_loss + F.cross_entropy(logits_proto_img2, labels_proto)

                kl_loss = kl_loss +  F.kl_div(F.log_softmax(logits_proto_modimg, dim=-1),
                                            F.softmax(logits_proto_img2, dim=-1),
                                            reduction="batchmean")

        loss['cls'] = loss['cls'] + proto_loss
        loss['kl_c'] = loss['kl_c'] + kl_loss     
        return loss

    
    def mask_constraint(self, mask1, mask2):
        mask = mask1 + mask2
        y = torch.ones_like(mask).float().cuda()
        return F.mse_loss(mask,y)

    def info_nce(self, query, target):
        x = torch.mm(query, target.T)
        labels = torch.arange(query.shape[0]).long().cuda()
        return F.cross_entropy(x * self.loss_T, labels)

    
    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl





import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import os
import pathlib
import copy

import artificery
import torchfields

from metroem.finetuner import optimize_pre_post_ups

def finetune_field(
    src,
    tgt,
    pred_res_start,
    src_defects=None,
    tgt_defects=None,
    lr=18e-1,
    sm=300e0,
    num_iter=60,
    crop=1
):
    mse_keys_to_apply = {
        'src': [
            {
                'name': 'src_defects',
                'binarization': {'strat': 'eq', 'value': 0},
                'coarsen_ranges': [(1, 0)]
             }
            ],
        'tgt':[
            {
                'name': 'tgt_defects',
                'binarization': {'strat': 'eq', 'value': 0},
                'coarsen_ranges': [(1, 0)]
            }
        ]
    }

    sm_keys_to_apply = {
       "src": [
         {
             "name": "src_defects",
             "mask_value": 1.0e-5,
             "binarization": {"strat": "eq", "value": 0},
             'coarsen_ranges': [(1, 0)]
         }
       ],
       "tgt": [
            {
                'name': 'tgt_defects',
                'binarization': {'strat': 'eq', 'value': 0},
                'coarsen_ranges': [(1, 0)]
             }
       ]

   }


    src_small_defects = None
    src_large_defects = None

    if src_defects is not None:
        src_defects = src_defects.squeeze(0)
    else:
        src_defects = torch.zeros_like(src)

    if src_small_defects is not None:
        src_small_defects = src_small_defects.squeeze(0)
    else:
        src_small_defects = torch.zeros_like(src)

    if src_large_defects is not None:
        src_large_defects = src_large_defects.squeeze(0)
    else:
        src_large_defects = torch.zeros_like(src)

    if tgt_defects is not None:
        tgt_defects = tgt_defects.squeeze(0)
    else:
        tgt_defects = torch.zeros_like(src_defects)
    pred_res_opt = optimize_pre_post_ups(
        src,
        tgt,
        pred_res_start,
        src_defects=src_defects,
        tgt_defects=tgt_defects,
        crop=crop,
        num_iter=num_iter,
        sm_keys_to_apply=sm_keys_to_apply,
        mse_keys_to_apply=mse_keys_to_apply,
        sm=sm,
        lr=lr,
        verbose=True,
    )
    return pred_res_opt

def create_model(checkpoint_folder, device='cpu', checkpoint_name="checkpoint"):
    a = artificery.Artificery()

    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)

    checkpoint_path = os.path.join(checkpoint_folder,
            f"{checkpoint_name}.state.pth.tar")
    if not os.path.isfile(checkpoint_path):
        print ("creating new checkpiont...")
        return my_p

    load_my_state_dict(my_p,
            torch.load(checkpoint_path,
                map_location=torch.device(device)))
    my_p.name = checkpoint_name

    return my_p


def load_my_state_dict(model, state_dict, delta=0):
    own_state = model.state_dict()

    reinit_downmodules = []
    reinit_upmodules = []

    for layer_name, param in state_dict.items():
        layer_name_deltad = copy.copy(layer_name)
        if "level_downmodules" in layer_name:
            prefix = "level_downmodules."
            delta = 0
        elif "level_upmodules" in layer_name:
            prefix = "level_upmodules."
            delta = 0
        else:
            prefix = None

        if prefix is not None:
            level = int(layer_name[len(prefix)][:1])
            level_d = level + delta
            layer_name_deltad = prefix + str(level_d) + layer_name[len(prefix) + 1:]

        load_weights = True
        if layer_name_deltad not in own_state:
            load_weights = False

        if load_weights:
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[layer_name_deltad].copy_(param)


class Aligner(nn.Module):
    def __init__(
        self,
        model_folder,
        pass_field=True,
        checkpoint_name="checkpoint",
        finetune=False,
        finetune_iter=100,
        finetune_lr=1e-1,
        finetune_sm=30e0,
        train=False,
        crop=1,
    ):
        super().__init__()

        this_folder = pathlib.Path(__file__).parent.absolute()
        self.net = create_model(model_folder, checkpoint_name=checkpoint_name)
        self.net.name = checkpoint_name
        self.finetune = finetune
        self.pass_field = pass_field
        self.finetune_iter = finetune_iter
        self.finetune_lr = finetune_lr
        self.finetune_sm = finetune_sm
        self.train = train
        self.crop = crop

    def forward(self, src_img, tgt_img, src_agg_field=None, tgt_agg_field=None,
            src_folds=None, tgt_folds=None,
            finetune=None,
            finetune_iter=None,
            finetune_lr=None,
            finetune_sm=None,
            train=None,
            return_state=False,
            final_stage=False,
            **kwargs):

        if 'cuda' in str(src_img.device):
            self.net = self.net.cuda(src_img.device)
        else:
            self.net = self.net.cpu()

        while len(src_img.shape) < 4:
            src_img = src_img.unsqueeze(0)
            tgt_img = tgt_img.unsqueeze(0)

        if src_agg_field is not None:
            while len(src_agg_field.shape) < 4:
                src_agg_field = src_agg_field.unsqueeze(0)

            src_agg_field = src_agg_field.field().from_pixels()
            warped_src_img = src_agg_field(src_img)
            src_agg_field = src_agg_field.pixels()
        else:
            warped_src_img = src_img

        if self.pass_field:
            net_input = torch.cat((src_img, tgt_img), 1).float()
        else:
            net_input = torch.cat((warped_src_img, tgt_img), 1).float()

        if (train is None and self.train) or train == True:
            pred_res = self.net.forward(x=net_input, in_field=src_agg_field)
        else:
            with torch.no_grad():
                pred_res = self.net.forward(x=net_input, in_field=src_agg_field)
                #print (pred_res.abs().mean())

        if not self.pass_field and src_agg_field is not None:
            pred_res = pred_res.field().from_pixels()(src_agg_field).pixels()

        if finetune or (finetune is None and self.finetune):
            if finetune_iter is None:
                finetune_iter = self.finetune_iter
            if finetune_lr is None:
                finetune_lr = self.finetune_lr
            if finetune_sm is None:
                finetune_sm = self.finetune_sm
                if final_stage:
                    finetune_sm *= 10.0e0
            embeddings = self.net.state['up']['0']['output']
            src_opt = embeddings[0, 1:embeddings.shape[1]//2].unsqueeze(0).detach()
            tgt_opt = embeddings[0, 1+embeddings.shape[1]//2:].unsqueeze(0).detach()

            src_defects = src_img == 0
            tgt_defects = tgt_img == 0
            #tgt_defects = None

            pred_res = finetune_field(
                src_opt,
                tgt_opt,
                pred_res,
                src_defects=src_defects,
                tgt_defects=tgt_defects,
                lr=finetune_lr,
                num_iter=finetune_iter,
                sm=finetune_sm,
                crop=self.crop,
            )
        if return_state:
            return pred_res.field(), self.net.state
        else:
            return pred_res.field()

    def get_embeddings(self, img, level=0, preserve_zeros=False):
        img = img.squeeze()
        assert len(img.shape) == 2
        while len(img.shape) < 4:
            img = img.unsqueeze(0)

        net_input = torch.cat((img, img), 1).float()

        with torch.no_grad():
            self.net.forward(x=net_input)

        emb = self.net.state['up'][str(level)]['output']
        img_emb = emb[0, 1:emb.shape[1]//2]
        if preserve_zeros:
            mask =  (img == 0).float()
            while mask.shape[-1] > img_emb.shape[-1]:
                mask = torch.nn.functional.max_pool2d(mask, 2)
            mask = mask != 0
            img_emb[..., mask.squeeze()] = 0
        return img_emb

    def save_checkpoint(self, checkpoint_folder):
        path = os.path.join(checkpoint_folder, f"{self.net.name}.state.pth.tar")
        torch.save(self.net.state_dict(), path)

    def load_checkpoint(self, checkpoint_folder, map_location):
        """Load from checkpoint

        Args:
            checkpoint_folder (str): path to checkpoint
            map_lcation (dict): {'cuda:0' : 'cudaN'}
        """
        checkpoint_path = os.path.join(checkpoint_folder,
                                       f"{self.net.name}.state.pth.tar")
        if  os.path.isfile(checkpoint_path):
            load_my_state_dict(self,
                               torch.load(checkpoint_path,
                                          map_location=map_location))


import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
from six import iteritems
import shelve
from scipy.ndimage.measurements import label
import h5py
import json
import sys 
import operator 
from scipy import ndimage

import copy
import os
from IPython.display import Image, display
from IPython.html.widgets import interact, fixed
from IPython.html import widgets
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

import os
from os.path import expanduser
import time

import torchvision
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.parallel import data_parallel
from torchvision import transforms

import random
import six

import importlib

import artificery
import metroem
importlib.reload(metroem)
import torchfields

from metroem import loss
from metroem import masks
from metroem import alignment
from metroem import aligner
from metroem import train
from metroem import helpers
from metroem import dataset

importlib.reload(helpers)
importlib.reload(aligner)
importlib.reload(alignment)
importlib.reload(masks)
importlib.reload(loss)
importlib.reload(train)
importlib.reload(dataset)

from metroem.alignment import align_sample
from metroem.loss import similarity_score, smoothness_penalty, get_dataset_loss, get_mse_and_smoothness_masks, smoothness_score
from metroem.helpers import reverse_dim, downsample

def visualize_residuals(res, figsize=(10,10), x_coords=None, y_coords=None, vec_grid=50):
    res = prepare_for_show(res)
    if res.shape[0] == 2:
        res = np.transpose(res, (1, 2, 0))

    assert res.shape[0] == res.shape[1]
    plt.figure(figsize=figsize)
    n = res.shape[0]
    y, x = np.mgrid[0:n, 0:n]
    
    if x_coords is None:
        x_coords = [0, res.shape[0]]
    if y_coords is None:
        y_coords = [0, res.shape[1]]
    
    ex = (1) * res[:, :, 0]
    ey = res[:, :, 1]
    r = np.arctan2(ex, ey)
    
    interval = (x_coords[1] - x_coords[0]) // vec_grid
    
    plt.quiver(  x[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],  
                 y[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ex[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], 
                ey[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], 
                 r[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], alpha=0.6)
    plt.quiver(x[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],  
                 y[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval],
                ex[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], 
                ey[x_coords[0]:x_coords[1]:interval, y_coords[0]:y_coords[1]:interval], edgecolor='k', facecolor='None', linewidth=.5)
    plt.gca().invert_yaxis()

def visualize_histogram(data, figsize=(10,10), x_coords=None, y_coords=None):
    pass
    
def get_np(pt):
    return pt.cpu().detach().numpy()


def prepare_for_show(img):
    if isinstance(img, torch.Tensor):
         img = get_np(img)
    img = img.squeeze()
    return img


rand_cmap = matplotlib.colors.ListedColormap(np.random.rand(256*32,3))

def display_image(img, x_coords=None, y_coords=None, normalize=False, figsize=(10, 10), mask=False, segmentation=False):
    if normalize and mask:
        raise Exception("Masks can't be normalized")
    img = prepare_for_show(img)

    if len(img.shape) == 3 and img.shape[-1] == 2:
        visualize_residuals(img, x_coords=x_coords, y_coords=y_coords)
        return
    
    plt.figure(figsize=figsize)
    
    if x_coords is None:
        x_coords = [0, img.shape[0]]
    if y_coords is None:
        y_coords = [0, img.shape[1]]
    
    if mask:
        plt.imshow(img[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap='gray')
    elif segmentation:
        cmap='gray'
        cmap = rand_cmap
        
        plt.imshow(img[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]].astype(np.int32), cmap=cmap)
    elif not normalize:
        plt.imshow(img[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap='gray', vmin=-1.5, vmax=1.5)
    else:
        plt.imshow(img[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]], cmap='gray')

        
def precompute_state(self, img_id, sup, val, state_file):
    self.compute_state(img_id, sup, val)
    try:
        
        state_h5 = h5py.File(state_file, 'w')
        img_shape = self.src.shape
        dset_shape = [0, 100, img_shape[3], img_shape[4]]
        if "main" not in state_h5:
            src_h5 = state_h5.create_dataset("main/src", dset_shape)
            tgt_h5 = state_h5.create_dataset("main/tgt", dset_shape)
            pred_tgt_h5 = state_h5.create_dataset("main/pred_tgt_h5", dset_shape)
        else:
            src_h5 = state_h5["main/src"]
            tgt_h5 = state_h5["main/tgt"]
            pred_tgt_h5 = state_h5["main/pred_tgt"]
        src_h5[:, img_id:img_id+1, :, :] = 0#self.src 
        tgt_h5[:, img_id:img_id+1, :, :] = 0#self.tgt
        pred_tgt_h5[:, img_id:img_id+1, :, :] = 0#self.pred_tgt_h5
        state_h5.close()
    except Exception as e:
        state_h5.close()
        raise e

def filter_folds(viz_mse_mask_var, input_mip=6):
    downsampler = nn.AvgPool2d(2) 
    for i in range(input_mip, 8):
        viz_mse_mask_var = downsampler(viz_mse_mask_var)
        
    viz_mse_mask_var = (viz_mse_mask_var < 0.3).type(torch.cuda.DoubleTensor)
                           
    viz_mse_mask_var = viz_mse_mask_var.type(torch.cuda.ByteTensor)
    mse_mask_np = get_np(viz_mse_mask_var.squeeze())
    filtered_mse_mask_np = filter_connected_component(mse_mask_np, 100) == 0
    return filtered_mse_mask_np

        
def segment_folds(fold_mask_np, memory_limit=None):
    initial_segments = get_initial_segments(fold_mask_np, memory_limit)
    segments  = cleanup_segment_stripes(initial_segments, n=16)
    return segments

def cleanup_segment_stripes(segments, n):
    seg_t = np.copy(segments.T)
    seg_start = np.zeros_like(seg_t[0])
    last_seg = np.zeros_like(seg_t[1])
    last_cleanup = 0
    
    for i in range(1, seg_t.shape[0]): 
        is_fold = seg_t[i] == 0
        was_fold = seg_t[i - 1] == 0
        last_seg_fold = last_seg == 0
        
        new_seg = (seg_t[i] != seg_t[i - 1])
        seg_width = i - seg_start
        
        bad_seg = new_seg * (seg_width < n) * (was_fold == False) * (is_fold == False)
        bad_seg_names = np.unique(seg_t[i - 1][bad_seg])
        
        if len(bad_seg_names) > 0:
            print (i, last_cleanup, bad_seg_names)
            cleanup_seg_id = i - 2
            ref_seg_id = i - 1

            ids_to_clean = None
            change_to_these = np.maximum(last_seg, new_seg)
            while (cleanup_seg_id >= 0):
                ids_to_clean = (seg_t[cleanup_seg_id] == seg_t[ref_seg_id]) * bad_seg * (last_seg_fold == False)
                if np.any(ids_to_clean) == False:
                    break
                seg_t[cleanup_seg_id][ids_to_clean] = change_to_these[ids_to_clean]
                cleanup_seg_id -= 1
            seg_t[ref_seg_id][bad_seg * (last_seg_fold == False)] = change_to_these[bad_seg * (last_seg_fold == False)]
            last_cleanup = i
                                  
        last_seg[new_seg] = seg_t[i - 1][new_seg]
        seg_start[new_seg] = i
    return seg_t.T

def get_initial_segments(fold_mask_np, memory_limit, smallest_seg):
    folds = fold_mask_np == False
    segments = np.zeros_like(folds, dtype=np.int32)
    last_fold = np.zeros_like(folds[0], dtype=np.int32)
    seg_start = np.zeros_like(folds[0], dtype=np.int32)
    
    segments[0] = 1
    segments[0][folds[0]] = 0
    max_in_line = np.copy(segments[0])
    
    
    for i in range(1, folds.shape[0]):
        
        
        is_fold = folds[i]
        was_fold = folds[i - 1]
        curr_seg = segments[i]
        last_seg = segments[i - 1]
        
        continue_ids = (is_fold == False) * (was_fold == False)
        short_stretch = (i - seg_start) < smallest_seg
        
        new_seg_ids = (is_fold == False) * (was_fold == True) * (short_stretch == False)
        jump_seg_ids = (is_fold == False) * (was_fold == True) * (short_stretch == True)
        
        curr_seg[continue_ids] = last_seg[continue_ids]
        curr_seg[new_seg_ids] = (max_in_line + 1)[new_seg_ids]
        
        fresh_start_ids = max_in_line == 0
        curr_seg[jump_seg_ids * (fresh_start_ids == False)] = max_in_line[jump_seg_ids * (fresh_start_ids == False)]
        curr_seg[jump_seg_ids * (fresh_start_ids == True)] = (max_in_line + 1)[jump_seg_ids * (fresh_start_ids == True)]
        
        seg_start[new_seg_ids] = i
        
        curr_seg[is_fold == True] = 0
        max_in_line = np.maximum(max_in_line, curr_seg)
        
        '''if memory_limit is not None:
            forgotten_folds = (i - last_fold) > memory_limit
            max_in_line[forgotten_folds] = np.maximum(1, max_in_line[forgotten_folds] - 1)'''
        last_fold[is_fold] = i
        
    return segments 

def expand_mask(mask_np, n=33):
    mask_var = torch.FloatTensor(mask_np.astype(np.float)).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones([1, 1, n, n])
    expanded_mask = torch.nn.functional.conv2d(mask_var, kernel, padding=n//2) 
    expanded_mask_np = get_np(expanded_mask[0, 0]) > 0
    return expanded_mask_np

def get_aligner(checkpoint_folder,
                checkpoint_name,
                train,
                finetune_lr,
                finetune_sm,
                finetune_iter,
                finetune):
    return aligner.Aligner(checkpoint_folder,
                            checkpoint_name=checkpoint_name,
                            train=train,
                            finetune_lr=finetune_lr,
                            finetune_sm=finetune_sm,
                            finetune_iter=finetune_iter,
                            finetune=finetune)

def get_dataset(data_dir, dataset_mip, stage, checkpoint_name, crop_mode=None):
    big_data = dataset.MultimipDataset(data_dir, field_tag=checkpoint_name)
    return big_data.get_train_dset(mip=dataset_mip,
                                    stage=stage,
                                    crop_mode=None,
                                    cropped_size=2048)


class PyramidVisualizer(object):
    def __init__(self, pyramid, dataset_mip, dataset):
        """Visualization manager

        Args:
            pyramid (metroem.aligner.Aligner)
            dataset_mip (int)
            dataset (metroem.dataset)
        """
        self.reverse = True
        self.fold_size = None
        self.src = None
        self.prev_id = -1
        self.prev_is_val = False
        self.prev_is_sup = False
        self.val_dataset = None
        self.train_dataset = None
        self.def_sup = True
        self.def_norm_img = True
        self.viz_mip = 0
        self.run_mip = 0
        self.sup = False
        self.prerun_augment = []
        
        self.pyramid = pyramid
        self.run_mip = 0
        self.viz_mip = 0
        
        self.dataset_mip = dataset_mip,
        self.sup_train_dataset = dataset
        #self.upsampler = nn.UpsamplingBilinear2d(scale_factor=2)   
        self.mse_keys_to_apply = {
        'src': [
            {'name': 'src_defects',
             'binarization': {'strat': 'eq', 'value': 0},
             "not_coarsen_ranges": [ (0,25)] },
            {'name': 'src',
             "not_coarsen_ranges": [(1, 0)],
             'binarization': {'strat': 'neq', 'value': 0}
             }
            ],
        'tgt':[
            {'name': 'tgt_defects',
             'binarization': {'strat': 'eq', 'value': 0},
             "coarsen_ranges": [(0, 0)]},
            {'name': 'tgt',
             "not_coarsen_ranges": [(10, 0)],
             'binarization': {'strat': 'neq', 'value': 0}
             }
        ]
    }
        self.sm_keys_to_apply = {                                                                                                 
           "src": [                                                                                                          
               {"name": "src_defects",                                                                                      
                "binarization": {"strat": "eq", "value": 0},                                                               
                "not_coarsen_ranges": [[1, 0.2], [4, 5.0]],                                                             
                "mask_value": 1.0e-5},                                                                                       
             {"name": "src",                                                                                             
                "fm": 0,                                                                                                  
                "binarization": {"strat": "neq", "value": 0}}                                                            
           ],                                                                                                              
           "tgt":[                                                                                                        
               {"name": "tgt_defects",                                                                                   
                "binarization": {"strat": "eq", "value": 0},                                                                 
                #"not_coarsen_ranges": [[1, 0.2], [2, 0.4], [3, 0.6], [4, 0.8]],                                                              
                "mask_value": 0.0e-5}                                                                                         
           ]                                                                                                              
       }
        
    
        
    def set_model(self, model):
        self.pyramid = model
        self.prev_id = -1
    
    def update_state(self, img_id, sup, val, state_file):
        if state_file is None:
            
            self.compute_state(img_id, sup, val)
        else:
            self.load_state(img_id, sup, val, state_file)
            
    def compute_state(self, img_id, sup, val):
        self.dataset = self.sup_train_dataset
                
        if self.prev_id == -1:
            self.clean_up()
            
        #make it refresh when sup is changed
        if self.prev_id == -1 or self.prev_is_sup != sup:
            self.prev_is_sup = sup
            self.prev_id = -1
            

        if img_id != self.prev_id or self.prev_is_val != val:
            self.prev_id = img_id
            self.prev_is_val = val
            #clean_sample = self.dataset[img_id][:, 2:-30, 2:-30].unsqueeze(0)
            
            raw_sample = copy.deepcopy(self.dataset[img_id])
           
            for aug in self.prerun_augment:
                if aug['type'] == 'contrast_half_src':
                    src = raw_sample[0, 0]
                    src[src.shape[0]//2:, :] *= aug['mult']
                elif aug['type'] == 'brightness_half_src':
                    src = raw_sample[0, 0]
                    src[src.shape[0]//2:, :] += aug['bump']
                else:
                    raise Exception("Unknown prerun transofm")
                    
            for k, v in six.iteritems(raw_sample):
                #print (v.shape)
                # why are we doing this
                raw_sample[k] = v.unsqueeze(0)  
                
            run_sample = copy.deepcopy(raw_sample)
            self.run_sample = run_sample
            for k, v in six.iteritems(run_sample):
                #print (v.shape)
                # why are we doing this: to have the same orientation as ng
                #run_sample[k] = v.permute(0, 2, 1)
                pass
            
            model_run_params = {"level_in": self.run_mip}
            
            run_result = align_sample(self.pyramid, run_sample)
            for k, v in six.iteritems(run_result):
                if hasattr(v, 'detach'):
                    run_result[k] = v.detach()
                
            run_src_var = run_result['src'].unsqueeze(0)                                                                   
            run_tgt_var = run_result['tgt'].unsqueeze(0)                                                                                                                             
            run_pred_res_var = run_result['pred_res']                                                                
            del run_result
            
            '''run_result = self.pyramid.run_pair(run_sample.unsqueeze(0), self.run_mip, reverse=self.reverse)
            run_src_var, run_tgt_var, run_true_res_var, run_pred_res_var, run_pred_tgt_var, run_masks_var = run_result
            
            del run_result, run_true_res_var, run_pred_tgt_var, run_masks_var'''
            
            #clean_smale = self.dataset[img_id][:, 2:-30, 2:-30].unsqueeze(0)
            viz_sample = copy.deepcopy(raw_sample)
                    
            viz_src_var = viz_sample['src']
            viz_tgt_var = viz_sample['tgt']
            
            if self.sup:
                viz_true_res_var = viz_sample['res']

            viz_src_var = viz_src_var
            viz_tgt_var = viz_tgt_var
            
            
            viz_pred_res_var = run_pred_res_var#.permute(0, 2, 1, 3).flip(3)
            
            viz_pred_res_var = viz_pred_res_var.squeeze()
            viz_sample['pred_res'] = viz_pred_res_var
            
            if 'src_field' in viz_sample:
                src_res_var = viz_sample['src_field'].field()
            else:
                src_res_var = torch.zeros_like(run_pred_res_var).field()
                
            viz_warped_src_var = src_res_var.from_pixels()(viz_src_var)
            viz_pred_tgt_var = viz_pred_res_var.from_pixels()(viz_src_var)
            
            viz_sample['pred_tgt'] = viz_pred_tgt_var


            viz_mse_mask_var, viz_smoothness_mask_var = get_mse_and_smoothness_masks(viz_sample, 
                                                                                     mse_keys_to_apply=self.mse_keys_to_apply,
                                                                                    sm_keys_to_apply=self.sm_keys_to_apply)
            
            
            self.viz_sample = viz_sample
            # Todo: visualize different penalties
            
            viz_rigidity_penalty_var = loss.rigidity(viz_pred_res_var.unsqueeze(0))
            viz_smoothness_penalty_var = viz_rigidity_penalty_var
            
            norm_tgt_var = torch.nn.InstanceNorm2d(1)(viz_tgt_var.unsqueeze(0))
            norm_pred_tgt_var = torch.nn.InstanceNorm2d(1)(viz_pred_tgt_var.unsqueeze(0))
            viz_masked_nrom_diff_var = viz_mse_mask_var * torch.abs(viz_tgt_var - viz_pred_tgt_var)
            del norm_tgt_var, norm_pred_tgt_var
            
            viz_diff_var = (viz_tgt_var - viz_pred_tgt_var)**2
            viz_masked_diff_var = viz_diff_var * viz_mse_mask_var
            viz_inv_res = torch.zeros_like(viz_pred_res_var)
            combined_identity_var = viz_inv_res
 
            self.img_dict = {
                'Source': {"img": get_np(viz_src_var), "norm": self.def_norm_img},
                'Warped Source': {"img": get_np(viz_warped_src_var), "norm": self.def_norm_img},
                'Target': {"img": get_np(viz_tgt_var), "norm": self.def_norm_img},
                'Predicted Target': {"img": get_np(viz_pred_tgt_var), "norm": self.def_norm_img},
                'Diff': {"img": get_np(viz_diff_var), "norm": self.def_norm_img},
                'Masked Diff': {"img": get_np(viz_masked_diff_var),
                                "norm": self.def_norm_img},
                'MSE Mask': {"img": get_np(viz_mse_mask_var)},
                'Masked Norm Diff': {"img": get_np(viz_masked_nrom_diff_var), "norm": self.def_norm_img},
                'Smoothness': {"img": get_np(viz_smoothness_penalty_var), "norm": self.def_norm_img},
                'Rigidity': {"img": get_np(viz_rigidity_penalty_var), "norm": self.def_norm_img},
                'Smoothness Mask': {"img": get_np(viz_smoothness_mask_var)},
                'Masked Smoothness': {"img": get_np(viz_smoothness_mask_var * viz_smoothness_penalty_var), 
                                      "norm": self.def_norm_img},
                'Masked Rigidity': {"img": get_np(viz_smoothness_mask_var * viz_rigidity_penalty_var), 
                                      "norm": self.def_norm_img},
            }

            self.vect_dict = {
                'Predicted Residual': get_np(viz_pred_res_var),
                'Source Residual': get_np(src_res_var),
                'Inverted Residual': get_np(viz_inv_res),
                'Combined': get_np(combined_identity_var)
            }
            
            self.hist_dict = {
                'Residual Histogram': get_np(viz_pred_res_var),
            }
            
            if self.sup:
                self.vect_dict['True Residual'] = get_np(viz_true_res_var),
                self.vect_dict['Residual Error'] =  get_np(viz_true_res_var - viz_pred_res_var)
    
    def clean_up(self):
        self.img_dict = {}
        self.vect_dict = {}
       
    def loadimg(self, val, sup, section_count, img_id, x_section, y_section, choice, state_file):
        self.update_state(img_id, sup, val, state_file)
        
        if choice in self.img_dict:
            normalize = ('norm' in self.img_dict[choice]) and (self.img_dict[choice]['norm'])
            is_mask = ('Mask ' in choice) or (' Mask' in choice)
            is_seg = ('segment' in choice.lower())
            img = copy.copy(prepare_for_show(self.img_dict[choice]['img']))
            
            x_section_size = img.shape[0] // section_count
            y_section_size = img.shape[1] // section_count
            
            x_coords = (x_section_size * x_section, x_section_size * (x_section + 1))
            y_coords = (y_section_size * y_section, y_section_size * (y_section + 1))

            if choice in ['Diff',  'Masked Diff']:
                print ('MSE: {}'.format(np.mean(np.abs(img[x_coords[0]:x_coords[1], y_coords[0]:y_coords[1]]))))
            display_image(img, mask=is_mask, segmentation=is_seg, normalize=normalize, x_coords=x_coords, y_coords=y_coords)
            
        elif choice in self.vect_dict:
            vecs = prepare_for_show(self.vect_dict[choice])

            x_section_size = vecs.shape[0] // section_count
            y_section_size = vecs.shape[1] // section_count
            
            x_coords = (x_section_size * x_section, x_section_size * (x_section + 1))
            y_coords = (y_section_size * y_section, y_section_size * (y_section + 1))
            
            
            visualize_residuals(vecs, x_coords=x_coords, y_coords=y_coords)

        elif choice in self.hist_dict:
            img = self.hist_dict[choice]
            x_section_size = img.shape[0] // section_count
            y_section_size = img.shape[1] // section_count
            
            x_coords = (x_section_size * x_section, x_section_size * (x_section + 1))
            y_coords = (y_section_size * y_section, y_section_size * (y_section + 1))
            visualize_histogram(img, x_coords=x_coords, y_coords=y_coords)
       
    def visualize(self, section_count=1, state_file=None, default_slice=9, default_x=0, default_y=0):
        
        self.id_selector = widgets.IntText(
            value=default_slice,
            description='Sample ID:',
            disabled=False
        )

        self.val_selector = widgets.Checkbox(
            value=False,
            description='Take from validation set',
            disabled=False
        )
        self.sup_selector = widgets.Checkbox(
            value=self.def_sup,
            description='Take from supervised dataset',
            disabled=False
        )
        
        self.section_count_selector = widgets.IntText(
            value=section_count,
            description='Section Count:',
            disabled=False
        )

        self.x_section_selector = widgets.IntText(
            value=default_x,
            description='X section:',
            disabled=False
        )
        self.y_section_selector = widgets.IntText(
            value=default_y,
            description='Y section:',
            disabled=False
        )
        
        buttons = ['Source', 'Warped Source', 'Target', 'Predicted Target', 
                   'Source Residual', 'Predicted Residual', 'Inverted Residual', 
                   'Combined',  'Diff', 'MSE Mask', 'Masked Diff', 'Residual Histogram',
                   'Smoothness Mask', 'Smoothness', 'Rigidity', 'Masked Smoothness', 'Masked Rigidity']

        # for supervised
        #buttons += ['True Residual', 'Error histogram', 'Vector histogram', 'Residual Error'],
        self.button_choice_1 = widgets.ToggleButtons(
            options=buttons,
            description='Choice:',
            disabled=False,
            button_style='',
        #     icons=['check'] * 3
        )
        interact(self.loadimg, img_id=self.id_selector, val=self.val_selector, sup=self.sup_selector, 
                 section_count=self.section_count_selector, x_section=self.x_section_selector, y_section=self.y_section_selector, 
                 choice=self.button_choice_1, state_file=fixed(state_file))


            
class simple_visualizer():
    def __init__(self):
        pass
    
    def display_multiimg(self, choice):
        i = self.names.index(choice)
        if isinstance(self.images[i], torch.Tensor):
            self.images[i] = self.images[i].cpu().detach().numpy()
        self.images[i].squeeze()
        
        plt.figure(figsize = (12,12))
        if self.crop == 0:
            plt.imshow(self.images[i], cmap='gray')
        else:
            plt.imshow(self.images[i][self.crop:-self.crop, self.crop:-self.crop], cmap='gray')
    
    def loadimg(self, choice, section_count, x_section, y_section):
        i = self.names.index(choice)
        if isinstance(self.images[i], torch.Tensor):
            self.images[i] = self.images[i].cpu().detach().numpy()
        self.images[i].squeeze()
        img = self.images[i].squeeze()
        img = prepare_for_show(img)
        x_section_size = img.shape[0] // section_count
        y_section_size = img.shape[1] // section_count

        x_coords = (x_section_size * x_section, x_section_size * (x_section + 1))
        y_coords = (y_section_size * y_section, y_section_size * (y_section + 1))
        
        display_image(img, normalize=True, x_coords=x_coords, y_coords=y_coords)
        
    def visualize(self, images, names=None, crop=0, section_count=1, x_section=0, y_section=0):
        if names is None:
            names = range(len(images))
        self.names = names
        self.images = images
        self.crop = crop
        
        button_choice = widgets.ToggleButtons(
                options=names,
                description='Image:',
                disabled=False,
                button_style='',
            )
        
        self.section_count_selector = widgets.IntText(
            value=section_count,
            description='Section Count:',
            disabled=False
        )

        self.x_section_selector = widgets.IntText(
            value=x_section,
            description='X section:',
            disabled=False
        )
        self.y_section_selector = widgets.IntText(
            value=y_section,
            description='Y section:',
            disabled=False
        )

        interact(self.loadimg, choice=button_choice, 
                 section_count=self.section_count_selector, 
                 x_section=self.x_section_selector, y_section=self.y_section_selector)



### 2nd cell ###
# import artificery
from torch.nn.parameter import Parameter

def load_my_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
             continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
            
def create_model(name, checkpoint_folder, checkpoint_init=False):                                                                                                                                                     
    a = artificery.Artificery(checkpoint_init=checkpoint_init)    
    
    spec_path = os.path.join(checkpoint_folder, "model_spec.json")
    my_p = a.parse(spec_path)                                                                                                                                                                

    checkpoint_path = os.path.join(checkpoint_folder, "{}.state.pth.tar".format(name))                                                                                                                     
    if os.path.isfile(checkpoint_path):         
        load_my_state_dict(my_p, torch.load(checkpoint_path))
        #my_p.load_state_dict(torch.load(checkpoint_path))                                                                                                                                                  
    my_p.name = name                                                                                                                                                                                       
    return my_p.cuda()  

def lap2(field):
    def dxf(f):
        p = Variable(torch.zeros((1,1,f.size(1),2))).cuda()
        return torch.cat((p, f[:,1:-1,:,:] - f[:,2:,:,:], p), 1)
    def dyf(f):
        p = Variable(torch.zeros((1,f.size(1),1,2))).cuda()
        return torch.cat((p, f[:,:,1:-1,:] - f[:,:,2:,:], p), 2)
    result = [field_dx(field), field_dy(field), dxf(field), dyf(field)]
    result = (sum(result) / 4.0) ** 2
    result = sum(torch.sum(result, -1))
    return result

def field_dx(f, forward=False):
    if forward:
        delta = f[:,1:-1,:,:] - f[:,2:,:,:]
    else:
        delta = f[:,1:-1,:,:] - f[:,:-2,:,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 0, 0, 1, 1, 0, 0))
    return result

def field_dy(f, forward=False):
    if forward:
        delta = f[:,:,1:-1,:] - f[:,:,2:,:]
    else:
        delta = f[:,:,1:-1,:] - f[:,:,:-2,:]
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 0, 0, 0, 0))
    return result

def field_dxy(f, forward=False):
    if forward:
        delta = f[:,1:-1,1:-1,:] - f[:,2:,2:,:]
    else:
        delta = f[:,1:-1,1:-1,:] - f[:,:-2,:-2,:]
        
    result = delta
    result = torch.nn.functional.pad(delta, pad=(0, 0, 1, 1, 1, 1, 0, 0))
    return result


def pix_identity(size, batch=1, device='cuda'):
    result = torch.zeros((batch, size, size, 2), device=device)
    x = torch.arange(size, device=device)
    result[:, :, :, 1] = x
    result = torch.transpose(result, 1, 2)
    result[:, :, :, 0] = x
    result = torch.transpose(result, 1, 2)
    return result
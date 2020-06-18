import sys
import json
import os
import time
import six
import argparse
import glob
import re

import modelhouse
import torch

from metroem import loss, augmentations
from metroem.alignment import aligner_train_loop
from metroem.dataset import MultimipDataset

from pdb import set_trace as st


#class MyDataParallel(torch.nn.DataParallel):
#    def __getattr__(self, name):
#        return getattr(self.module, name)

def get_pyramid_modules(pyramid_path):
    module_dict = {}
    contents = next(os.walk(pyramid_path))[1]

    for m in contents:
        pattern = r"(\d)_mip(\d)in_(.*)"
        match = re.search(pattern, m)
        if match is None:
            raise Exception(f"Invalid member in the pyramid: {m}.\n"\
                    "Pyramid folder must only contain model folders with names in " \
                    "format '{0-9}_mip{0-9}in_{arbitrary name}'")
        module_id = int(match.group(1))
        module_mip_in = int(match.group(2))
        module_path = os.path.join(pyramid_path, m)
        module_dict[module_id] = {"path": module_path, "mip_in": module_mip_in}

    return module_dict


def train_module(model, train_params, train_dset, val_dset,
        checkpoint_path, aug_params=None):
    assert aug_params is None

    val_data_loader = torch.utils.data.DataLoader(val_dset, batch_size=1, shuffle=True,
            num_workers=0, pin_memory=False)

    trainable = []
    trainable.extend(list(model.parameters()))

    for epoch_params in train_params:
        smoothness = epoch_params["smoothness"]
        if "print_every" in epoch_params:
            print_every = epoch_params["print_every"]
        else:
            print_every = None

        if "num_sample" in epoch_params:
            num_samples = epoch_params["num_samples"]
        else:
            num_samples = 10000000

        lr = epoch_params["lr"]
        num_epochs = epoch_params["num_epochs"]
        mse_keys_to_apply = epoch_params["mse_keys_to_apply"]
        sm_keys_to_apply = epoch_params["sm_keys_to_apply"]
        loss_spec = epoch_params["loss_spec"]
        loss_type = epoch_params["loss_spec"]["type"]

        simple_loss = loss.unsupervised_loss(smoothness, use_defect_mask=True,
                sm_keys_to_apply=sm_keys_to_apply, mse_keys_to_apply=mse_keys_to_apply)
        if loss_type == "plain":
            training_loss = simple_loss
        elif loss_type == "metric":
            training_loss = loss.multilevel_metric_loss(
                    loss_fn=simple_loss,
                    mip_in=0,
                    **loss_spec['params']
                    )

        augmentor = None
        if "augmentations" in epoch_params:
            augmentor = augmentations.Augmentor(epoch_params["augmentations"])

        train_dset.set_size_limit(num_samples)
        train_data_loader = torch.utils.data.DataLoader(train_dset, batch_size=1, shuffle=True,
                num_workers=0, pin_memory=False)

        optimizer = torch.optim.Adam(trainable, lr=lr, weight_decay=0)
        aligner_train_loop(model, 0, train_data_loader, val_data_loader, optimizer,
                num_epochs=num_epochs, loss_fn=training_loss,
                print_every=print_every, checkpoint_folder=checkpoint_path,
                augmentor=augmentor)

    pass

def train_pyramid(pyramid_path, dataset_path, train_stages, checkpoint_name,
        generate_field_stages, train_params=None, aug_params=None):
    pyramid_path = os.path.expanduser(pyramid_path)
    module_dict = get_pyramid_modules(pyramid_path)
    print (f"Loading dataset {dataset_path}...")
    dataset = MultimipDataset(dataset_path, aug_params, field_tag=checkpoint_name)

    prev_mip = None
    for stage in sorted(module_dict.keys()):
        module_mip = module_dict[stage]["mip_in"]
        module_path = module_dict[stage]["path"]
        model = None

        if train_stages is None or stage in train_stages:
            print (f"Training module {stage}...")
            model = modelhouse.load_model_simple(module_path,
                                                 finetune=False,
                                                 pass_field=True,
                                                 checkpoint_name=checkpoint_name)
            model
            if str(module_mip) not in train_params:
                raise Exception(f"Training parameters not specified for mip {module_mip}")

            mip_train_params = train_params[str(module_mip)]
            train_module(model, train_params=mip_train_params,
                    train_dset=dataset.get_train_dset(mip=module_mip, stage=stage),
                    val_dset=dataset.get_val_dset(mip=module_mip, stage=stage),
                    checkpoint_path=os.path.join(module_path, "model"))

            print (f"Done training module {stage}!")

        if generate_field_stages is None or stage in generate_field_stages:
            print (f"Generating fields with module {stage}...")
            model = modelhouse.load_model_simple(module_path,
                    finetune=True,
                    pass_field=True,
                    finetune_iter=300//(2**stage),
                    checkpoint_name=checkpoint_name)
            dataset.generate_fields(model, mip=module_mip, stage=stage)
            print (f"Done generating fields with module {stage}...")


def main():
    parser = argparse.ArgumentParser(description='Train SeamLESS.')

    parser.add_argument('--pyramid_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint")
    parser.add_argument('--train_mode', choices=['scratch', 'finetune'], type=str.lower, default='scratch')
    parser.add_argument('--train_params_path', type=str, default=None)
    parser.add_argument('--gpu', type=str, default="0")
    parser.add_argument('--train_stages', type=int, default=None, nargs='+')
    parser.add_argument('--generate_field_stages', type=int, default=None, nargs='+')

    parser.add_argument('--no_redirect_stdout', dest='redirect_stdout',
                        action='store_false')
    parser.set_defaults(redirect_stdout=True)
    args = parser.parse_args()

    if args.train_mode == 'custom':
        params_path = args.train_params_path
    else:
        assert args.train_params_path is None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        params_path = os.path.join(dir_path, f"params/{args.train_mode}.json")

    with open(params_path) as f:
        train_params = json.load(f)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= args.gpu

    if args.redirect_stdout:
        log_path = os.path.join(args.pyramid_path, f"{args.checkpoint_name}.log")
        log_file = open(log_path, 'a')
        sys.stdout = log_file


    train_pyramid(pyramid_path=args.pyramid_path,
                  dataset_path=args.dataset_path,
                  train_stages=args.train_stages,
                  generate_field_stages=args.generate_field_stages,
                  train_params=train_params,
                  checkpoint_name=args.checkpoint_name)

if __name__ == "__main__":
    main()

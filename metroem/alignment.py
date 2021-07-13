import torch
import numpy as np
import copy
import sys
import time

import torch.distributed as dist

from metroem import loss

from pdb import set_trace as st

def align_sample(model, bundle, train=False):
    tgt = bundle['tgt'].unsqueeze(1)
    src = bundle['src'].unsqueeze(1)

    src_field = None
    if 'src_field' in bundle:
        src_field = bundle['src_field']
    pred_res, state = model.forward(src_img=src, tgt_img=tgt, src_agg_field=src_field,
            return_state=True, train=train)

    bundle['pred_tgt'] = pred_res.from_pixels()(src)
    bundle['pred_res'] = pred_res

    bundle['state'] = state
    #print (pred_res[..., (src != 0).squeeze()].abs().mean())
    return bundle


def aligner_train_loop(rank,
                       model,
                       mip_in,
                       train_loader,
                       val_loader,
                       optimizer,
                       print_every=50,
                       num_epochs=1000,
                       loss_fn=None,
                       reverse=True,
                       manual_run_transform_def={},
                       manual_loss_transform_def={},
                       checkpoint_folder='./checkpoints',
                       loss_from_state=False,
                       augmentor=None):
    times = []
    count = 0
    running_loss = 0.0
    if loss_fn == None:
        loss_fn = model.loss

    if rank == 0:
        aligner_validate_and_save(val_loader,
                                  model,
                                  loss_fn,
                                  mip_in,
                                  0,
                                  0,
                                  [0],
                                  reverse,
                                  checkpoint_folder=checkpoint_folder)
    for epoch in range(num_epochs):
        count = 0
        s = time.time()
        for bundle in train_loader:
            transform_seed = np.random.randint(10000000)
            np.random.seed(transform_seed)

            if augmentor is not None:
                bundle = augmentor(bundle)

            raw_bundle = copy.deepcopy(bundle)
            run_bundle = raw_bundle
            #run_bundle = apply_transform(raw_bundle, run_transform)

            run_bundle = align_sample(model, run_bundle, train=True)

            np.random.seed(transform_seed)
            raw_bundle = copy.deepcopy(bundle)
            loss_bundle = raw_bundle
            #loss_bundle = apply_transform(raw_bundle, loss_transform)
            loss_bundle['state'] = run_bundle['state']
            loss_bundle['pred_res'] = run_bundle['pred_res'].field()

            loss_bundle['pred_tgt'] = loss_bundle['pred_res'].from_pixels()(loss_bundle['src'])

            loss_dict = loss_fn(loss_bundle)
            loss_var = loss_dict['result']
            running_loss += loss_var.cpu().detach().numpy()
            if loss_var is None or loss_var != loss_var or \
                    loss_var > 1E3:
                print (f"Bad loss for sample #{bundle['id']}: {loss_var}")
                import pdb; pdb.set_trace()
                continue

            optimizer.zero_grad()
            loss_var.backward()
            good_grad = True
            for p in model.parameters():
                if p.grad is not None and p.grad.mean() != p.grad.mean():
                    good_grad = False
                    print ("Bad grad", loss_var)
                    break
            if not good_grad:
                print ("Bad grad", loss_var)
                continue

            avg = 0
            optimizer.step()

            e = time.time()
            times.append(e - s)

            count += 1

            # if print_every is not None and ((count + 1) % print_every == 0):
            #     train_loss = running_loss / count
            #     aligner_validate_and_save(val_loader, model, loss_fn, mip_in, epoch, train_loss,
            #             times, reverse,
            #             checkpoint_folder=checkpoint_folder)

            #     running_loss = 0.0
            s = time.time()

        if (print_every is None) and (rank == 0):
            train_loss = running_loss / count
            aligner_validate_and_save(val_loader, model, loss_fn, mip_in, epoch, train_loss,
                    times, reverse,
                    checkpoint_folder=checkpoint_folder)
            times = []
            running_loss = 0.0
        # dist.barrier()
        # # synchronize processes from same checkpoint
        # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        # model.module.aligner.load_checkpoint(checkpoint_folder, map_location=map_location)


def aligner_validate_and_save(val_loader, model, loss_fn, mip_in, epoch, train_loss, times, reverse,
                      checkpoint_folder):
    with torch.set_grad_enabled(False):
        np.random.seed(5566)
        if not val_loader is None:
            val_loss_fn = loss_fn
            loss_dict = loss.get_dataset_loss(model, val_loader, loss_fn=val_loss_fn, mip_in=mip_in, reverse=reverse)
            val_loss = loss_dict['result']
            vec_magn = loss_dict['vec_magnitude']
            smooth_loss = loss_dict['smoothness']
            sim_loss = loss_dict['similarity']
            val_report = 'Val: {:.2E}, Vec M: {:.2E}, Sm: {:.2E}, Sim: {:.2E}'.format(val_loss, vec_magn, smooth_loss, sim_loss)
        else:
            val_report = ''

        np.random.seed()
        print ("Epoch {}: Tra: {:.2E}, {}, T: {:.4}sec".format(epoch, float(train_loss), val_report, np.mean(times)))

        sys.stdout.flush()
        model.module.aligner.save_checkpoint(checkpoint_folder=checkpoint_folder)

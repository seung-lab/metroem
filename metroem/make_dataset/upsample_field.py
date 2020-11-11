import torch
import torchfields
import numpy as np
import argparse
import h5py

def upsample_field(src_dset, dst_dset, src_mip, dst_mip, device='cuda'):
    """Upsample a field datset from SRC_MIP to DST_MIP

    Args:
        src_dset (h5py.dset)
        dst_dset (h5py.dset)
        src_mip (int): MIP level of the input dataset
        dst_mip (int): MIP level of the output dataset
    """
    assert(src_mip > dst_mip)
    with torch.no_grad():
        dst_size = dst_dset.shape[-1]
        for n in range(src_dset.shape[0]):
            print('Upsampling {} / {}'.format(src_dset.shape[0], n))
            src_field = torch.tensor(src_dset[n:n+1], device=device).field()
            src_field = src_field * (2**src_mip)
            dst_field = src_field.up(mips=src_mip - dst_mip)
            dst_field = dst_field / (2**dst_mip)
            dst_field_cropped = dst_field[0, :, :dst_size, :dst_size]
            if dst_field_cropped.is_cuda:
                dst_field_cropped = dst_field_cropped.cpu()
            dst_dset[n] = dst_field_cropped.numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Upsample H5 Field')
    parser.add_argument('--src_h5', type=str,
            help='Path to H5 file, with dataset stored in main')
    parser.add_argument('--dst_h5',  type=str,
            help='Path to H5 file which will be upsampled')
    parser.add_argument('--src_mip',  type=int)
    parser.add_argument('--dst_mip',  type=int)
    parser.add_argument('--cuda',  action='store_true')
    args = parser.parse_args()

    src_dset = h5py.File(args.src_h5, 'r')['main']
    shape = src_dset.shape
    dst_h5 = h5py.File(args.dst_h5, 'w')
    field_shape = (shape[0], 2, shape[-2], shape[-1])
    chunks = (1, 2, shape[-2], shape[-1])
    dst_dset = dst_h5.create_dataset("main", 
                                        shape=field_shape,
                                        dtype=np.float32,
                                        chunks=chunks,
                                        compression='lzf',
                                        scaleoffset=2
                                        )
    device = 'cuda' if args.cuda else 'cpu'
    upsample_field(src_dset=src_dset,
                   dst_dset=dst_dset,
                   src_mip=args.src_mip,
                   dst_mip=args.dst_mip,
                   device=device)


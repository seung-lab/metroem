import torch
import torchfields
import numpy as np
import argparse
import h5py
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, Executor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description='Upsample H5 Field')
    parser.add_argument('--src_path', type=str,
            help='Path to H5 file, with dataset stored in main')
    parser.add_argument('--dst_path',  type=str,
            help='Path to H5 file which will be upsampled')
    parser.add_argument('--src_mip',  type=int)
    parser.add_argument('--dst_mip',  type=int)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--cuda',  action='store_true')
    args = parser.parse_args()

    src_h5 = h5py.File(args.src_path, 'r')
    src_dset = src_h5['main']
    shape = src_dset.shape
    # src_h5.close()
    # dst_h5.close()
    device = 'cuda' if args.cuda else 'cpu'
    assert(args.src_mip > args.dst_mip)
    
    n_size = shape[0] // args.parallel
    n_starts = list(range(0,shape[0],n_size))
    n_stops = n_starts[1:] + [shape[0]]
    upsample_args = [(args.src_path, 
                      args.src_mip, 
                      args.dst_mip, 
                      n_start, 
                      n_stop,
                      device) for (n_start, n_stop) in zip(n_starts, n_stops)]

    def upsample_field(upargs):
        """Upsample a field datset from SRC_MIP to DST_MIP

        Args:
            src_path (str)
            src_mip (int)
            dst_mip (int)
            n_start (int): index
            n_stop (int): index
            device (str)
        """
        print(upargs)
        src_path, src_mip, dst_mip, n_start, n_stop, device = upargs
        src_path = Path(src_path)
        src_h5 = h5py.File(src_path, 'r')
        src_dset = src_h5['main']
        dst_path = src_path.parent / 'tmp_{}_{}_MIP{}.h5'.format(n_start, n_stop, dst_mip)
        dst_h5 = h5py.File(dst_path, 'w')
        shape = src_dset.shape
        dst_shape = (n_stop-n_start+1, 2, shape[-2], shape[-1])
        chunks = (1, 2, dst_shape[-2], dst_shape[-1])
        dst_dset = dst_h5.create_dataset("main", 
                                            shape=dst_shape,
                                            dtype=np.float32,
                                            chunks=chunks,
                                            compression='lzf',
                                            scaleoffset=2
                                            )
        with torch.no_grad():
            for n in range(n_start, n_stop):
                print('Upsampling {} / {}'.format(shape[0], n))
                src_field = torch.tensor(src_dset[n:n+1], device=device).field()
                src_field = src_field * (2**src_mip)
                dst_field = src_field.up(mips=src_mip - dst_mip)
                dst_field = dst_field / (2**dst_mip)
                dst_field_cropped = dst_field[0, :, :dst_shape[-2], :dst_shape[-1]]
                if dst_field_cropped.is_cuda:
                    dst_field_cropped = dst_field_cropped.cpu()
                dst_dset[n-n_start] = dst_field_cropped.numpy()
        return n_start, n_stop

    with ProcessPoolExecutor(max_workers=args.parallel) as executor:
        # executor.map(lambda x: upsample_field(*x), upsample_args)
        results = executor.map(upsample_field, upsample_args)

    print(len(list(results)))



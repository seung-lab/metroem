import os
import argparse
import h5py
from pathlib import Path
import numpy as np

def concat_shards(temp_dir, dst_path, inc=1000):
    temp = Path(temp_dir)
    paths = {}
    for fn in temp.iterdir():
        if fn.suffix == '.h5':
            k = int(fn.name.split('_')[0])
            assert(k not in paths)
            paths[k] = fn
    if len(paths) == 0:
        return None
    sorted_keys = sorted(paths.keys())
    for k in sorted_keys:
        print(paths[k])
    dsets = [h5py.File(paths[k], 'r')['main'] for k in sorted_keys]
    shapes = [d.shape for d in dsets]
    # check if all shards are the same size
    for i in range(1,len(shapes[0])):
        assert(all([s[i] == shapes[0][i] for s in shapes]))
    n = sum([s[0] for s in shapes])
    shape = (n, *shapes[0][1:])
    field = h5py.File(dst_path, 'w')
    chunks = (1, *shapes[0][1:])
    field_dset = field.create_dataset("main", 
                                      shape=shape,
                                      dtype=np.float32,
                                      chunks=chunks,
                                      compression='lzf',
                                      scaleoffset=2)
    k_start = 0 
    for dset in dsets:
        d_starts = list(range(0, dset.shape[0], inc))
        d_stops = d_starts[1:] + [dset.shape[0]]
        for d_start, d_stop in zip(d_starts, d_stops):
            d_inc = d_stop - d_start
            k_stop = k_start + d_inc
            print('Concatenating {}:{}'.format(k_start, k_stop))
            field_dset[k_start:k_stop] = dset[d_start:d_stop]
            k_start = k_stop
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Concatenate field shards into a dataset.')

    parser.add_argument('--temp_dir', 
            type=str, 
            help='Path to dir with shards from generate_fields')
    parser.add_argument('--dst_path', 
            type=str,
            help='Path to output H5')
    parser.add_argument('--increment', type=int, default=1000)
    args = parser.parse_args()

    assert(os.path.exists(args.temp_dir))
    assert(not os.path.exists(args.dst_path))

    concat_shards(temp_dir=args.temp_dir, 
                  dst_path=args.dst_path, 
                  inc=args.increment)


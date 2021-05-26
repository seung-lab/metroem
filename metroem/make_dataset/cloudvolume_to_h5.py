import os
import argparse
import h5py
from cloudvolume import CloudVolume
from pathlib import Path
import numpy as np

def cloudvolume_to_h5(cv_path, dst_path):
    vol = CloudVolume(cv_path)
    shape = (vol.shape[2], 2, vol.shape[0], vol.shape[1])
    chunks = (1, 2, vol.shape[0], vol.shape[1])
    field = h5py.File(dst_path, 'w')
    dset = field.create_dataset("main", 
                              shape=shape,
                              dtype=np.float32,
                              chunks=chunks,
                              compression='lzf',
                              scaleoffset=2)
    for k in range(vol.shape[2]):
        dset[k:k+1] = vol[:,:,k:k+1][:,:,0,:].transpose(2,0,1)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create H5 file from CloudVolume')

    parser.add_argument('--cv_path', 
            type=str, 
            help='Path to dir with shards from generate_fields')
    parser.add_argument('--dst_path', 
            type=str,
            help='Path to output H5')
    args = parser.parse_args()

    assert(not os.path.exists(args.dst_path))

    cloudvolume_to_h5(cv_path=args.cv_path, 
                      dst_path=args.dst_path)


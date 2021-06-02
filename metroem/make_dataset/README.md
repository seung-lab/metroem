# How to curate a dataset  
Properly curating a dataset is very important.
The scripts in [`download_image.py`](../download_image.py)
provide a method to download a set of section pairs 
within the same bounding box that's been shifted in z, but 
sometimes we want a more specific set of sample.

To do this, we create a spec file, which lists
the center MIP0 location of each source & target image pair that we want
to include in our dataset.
See [`spec_template.json`](spec_template.json) for the 
spec convention.
The pairs list uses the following convention:
```
"pairs": [
    {
        "src": {
            "x":
            "y":
            "z":
        },
        "tgt": {
            "x":
            "y":
            "z":
        }, 
    },
    ...
]
```

## How to create a spec file
We can create a spec file from scratch, but far easier is
to use neuroglancer to collect a set of points the indicate
where we would like to draw samples.

### Collect points in neuroglancer  
Use a single annotation layer in neuroglancer to collect a
set of points that mark the bounding box start for a source image.
Export that set of annotations as a csv file.

### Create spec file from neuroglancer points
Using the neuroglancer points, we can create a spec file with
[`create_spec.py`](create_spec.py). 
Here is an example script which creates `(src,tgt)` pairs
for `(z,z-1)`, as well as for `(z+offset,z+offset-1)`:
```
python create_spec.py \
--src_spec_path ./spec_template.json \
--dst_spec_path [SPEC_PATH] \
--points_path [PATH_TO_NEUROGLANCER_ANNOTATIONS_CSV] \
--offsets -1 -5 -10 -15
```

#### What does the spec template mean?
When creating a dataset, the images must be normalized and masked, the
target image must be warped by any previous fields, and the source image
must have an associated previous field that will be introduced at the
beginning of the decoding stage of the model.

The final dataset will be compiled at the `dst_mip` with size `x_size`,`y_size`
in the root of the spec. The points listed in `pairs` are defined in MIP0 pixels. 
The `max_mip` is used during `create_spec.py` to snap any points from 
neuroglancer to pixels at this resolution (so all higher-resolution images will 
be integer pixel aligned).

Under `image`, the `src_mip` indicates at which resolution to pull the image.
The `reference` object is used to download an image from which the mean and
standard deviation will be calculated for normalizing the final image.

The `field` object is similar to the `image` object, but in this case, the
`reference` object defines a field which will be profiled to determine the
appropriate offset for adjusting the bounding box. Because the images
and fields in a metroem dataset are the same size, we must remove any
translational bias from a previous field. If we didn't, that previous field
may require the image data to be much larger. Profiling the field is a 
common step in all of our alignment processing.

The `mask` object lists a series of masks, which will pull images as their
`src_mip` resolution, interpolate them to the `dst_mip` in the root of the
spec, then produce a 1-hot map for any pixels greater than the threshold
(we're working on other comparators). Any pixel in this mask will be set
to 0 in the final image.

#### How do we locate the same region at different resolutions?
In metroem, we currently depend on coordinates in the spec to be snapped
to voxels at the `max_mip`, as well as sizes of images to be larger than
`2^(highest mip - lowest mip)` (e.g. 1024 is typically adequate). For this
reason, it's less error prone to use `create_spec.py` rather than manually
adding points.

If a `field` is being used, it's important that the `reference` object be
identical for each resolution being downloaded. This ensures that the same
region is being profiled to determine the correct offset for sampling the
image.

# Use spec file to download data
We can download our dataset, once we have a spec file.
Here's an example script that downloads a dataset's image
and field H5 files, given a dataset spec file:
```
python make_dataset.py \
--spec_path [SPEC_PATH] \
--dst_path [PATH_TO_DATASET_DIR] \
--parallel 4
```

## Generating fields over a dataset
If we are training multiple aligners, we must use the 
coarser aligners to generate a paired set of fields for the
image dataset that will be used to train a finer aligner.
Generating fields across a large dataset can take time.
Here's an example script to generate those fields into a CloudVolume,
using multiple GPUs:
```
python generate_fields.py \
--pyramid_path [PATH_TO_COARSE_ALIGNER] \
--image_path [PATH_TO_COARSE_DATASET] \
--prev_field_path [PATH_TO_COARSE_ALIGNER_INPUT_FIELDS] \
--temp_dir [PATH_TO_COARSE_ALIGNER_OUTPUT_FIELD_SHARDS] \
--stage 1 \
--checkpoint_name test \
--src_mip 6 \
--dst_mip 4 \
--gpu 0,1,2,3
```
Then by compiling those CloudVolumes into H5 files for a metroem dataset:
```
python cloudvolume_to_h5.py \
--cv_path [PATH_TO_COARSE_ALIGNER_OUTPUT_FIELD_CLOUDVOLUME] \
--dst_path [FINAL_DATASET_FILEPATH] 
```

The `generate_fields.py` will produce two CloudVolumes, one for the `src_mip`,
and one for the `dst_mip`. 

### Note on upsampling with pytorch
Our image MIP stacks are based on factors of 2. In metroem, we upsample directly 
between resolutions with bilinear interpolation, skipping any intermediary
MIP levels. This is similar but not equivalent to upsampling iteratively by 
factors of 2.

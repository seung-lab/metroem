# How to curate a dataset  
Properly curating a dataset is very important.
The scripts in [`download_image.py`](../download_image.py)
provide a method to download a stretch of data, but 
sometimes we want a more specific set of sample.

To do this, we create a spec file, which details
the exact location of each source & target image pair we want
to include in our dataset.
See [`spec_template.json`](spec_template.json) for the 
spec convention.
The pairs list uses the following convention:
```
"pairs": [
    {
        "src": {
            "x_start":
            "y_start":
            "z_start":
        },
        "tgt": {
            "x_start":
            "y_start":
            "z_start":
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
Here is an example script which creates src,tgt pairs,
as well as tgt,src pairs, where the pairs have offsets
`-1, -2, -3`:
```
python create_spec.py \
--src_spec_path ./spec_template.json \
--dst_spec_path [SPEC_PATH] \
--points_path [PATH_TO_NEUROGLANCER_ANNOTATIONS_CSV] \
--offsets -1 -2 -3 \
--permute_pairs
```

## Use spec file to download data
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
Here's an example script to generate those fields, by first
producing it in subsets across multiple GPUs:
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
Then by concatenating them into one final dataset:
```
python concat_fields.py \
--temp_dir [PATH_TO_COARSE_ALIGNER_OUTPUT_FIELD_SHARDS] \
--dst_path [FINAL_DATASET_FILEPATH] 
```
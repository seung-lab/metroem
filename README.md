# MetroEM
Train all aligners

# Get started
MetroEM is based on modelhouse, artificery, and scalenet packages.
It's useful to understand these packages before proceeding.

## Create a model
Create a directory, with subdirectories for each module.
It's easiest to adapt an existing model.
For example, use `modelhouse` to download the following model:
```
modelhouse load gs://corgie/models/pyramid_m4m6m9
```

### Module conventions  
Module subdirectories must follow the name convention
```
<STAGE>_mip<IN_MIP>in_<SUFFIX>
```
such that:
* `STAGE` (int): processing order of module (lowest-to-highest, 0-indexed)
* `IN_MIP` (int): mip level of the images to be used as input
* `SUFFIX` (str): string for description only  

Each module subdirectory must follow `modelhouse` directory conventions.

## Create a training set
Training sets consists of an image dataset as well as a field dataset.  

An image dataset is a set of image pairs (source and target images),
one set of image pairs for the MIP level required as input for
each module.

A field dataset is a displacement field computed by a previous module,
one field for each image in an image set.

## Create an image dataset
Image datasets may be created from CloudVolumes.
For models with multiple modules, the image set of each module
must be aligned in physical space across the span of MIP levels.  

To create a dataset from a CloudVolume, use `download_image.py`.
For example,
```
python download_image.py \
--dst_folder  ~/data \
--z_start 8175 \
--z_end 8180 \
--x_offset 200960 \
--y_offset 136448 \
--mip 3 5 \
--patch_size 1536 1536 \
--cv_path <CloudVolume path>
```
  
### Create a field dataset 
Field datasets will be automatically created during the training procedure.

#### Downloading from an existing field CloudVolume  
There may be times when a user would like to create a field dataset from
an existing field CloudVolume.
To do so, use `download_field.py`.
For example:
```
python download_field.py \
--dst_folder  ~/data/ \ 
--z_start 8175 \       
--z_end 8180 \	
--x_offset 200960 \
--y_offset 136448 \
--src_mip 8 \
--dst_mip 5 \
--patch_size 1536 \
--suffix precomputed \
--cv_path <CloudVolume path>
```

Once downloaded, a field dataset must be used as input when creating a new
image dataset. For example:
```
python download_image.py \
--dst_folder  ~/data \
--z_start 8175 \
--z_end 8180 \
--x_offset 200960 \
--y_offset 136448 \
--mip 5 \
--patch_size 1536 \
--cv_path <CloudVolume path> \
--field_dset ~/data/field_0_x200960_y136448_z8175_MIP5_precomputed.h5 \
```

Each downloaded field is adjusted by its average translation, and the 
bounding box of the associated image must also be adjusted by the 
translation.

## Train
With a model & dataset in place, it's time to train the model with `train.py`.
For example:
```
python train.py \
--pyramid_path ~/model/m4m6m8 \
--dataset_path ~/data \
--train_mode scratch \
--train_stages 1 \
--generate_field_stages 1 \
--checkpoint_name precomputed \
--no_redirect_stdout
```

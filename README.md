# mosaic-preprocessing

<p align="center">
<img src="https://github.com/blahner/mosaic-preprocessing/blob/main/assets/mosaic_logo.png" width="50%" alt="MOSAIC Logo">
</p>
    
This repository serves two purposes: (1) share the preprocessing code for the eight datasets used in the original MOSAIC publication and (2) provide a template repository for others to preprocess their fMRI dataset to be MOSAIC-compliant. A public GitHub repository that includes all preprocessing scripts, like this repository, is required to make your fMRI dataset MOSAIC-compliant.

## MOSAIC preprocessing for the MOSAIC [manuscript](TODO)
This repository details the preprocessing for the following datasets:

- [BOLD5000](https://openneuro.org/datasets/ds001499)
- [BOLD Moments Dataset (BMD)](https://openneuro.org/datasets/ds005165)
- [Generic Object Decoding (GOD)](https://openneuro.org/datasets/ds001246)
- [Deeprecon](https://openneuro.org/datasets/ds001506)
- [Human Actions Dataset (HAD)](https://openneuro.org/datasets/ds004488)
- [THINGS](https://openneuro.org/datasets/ds004192)
- [Natural Object Dataset (NOD)](https://openneuro.org/datasets/ds004496)
- [Natural Scenes Dataset (NSD)](https://registry.opendata.aws/nsd/)

The steps are preprocessing stimulus set, fMRI data, and validation. The goal of this first part is to enable others to reproduce your preprocessing.
First, create an environment.

```
conda create -n mosaic-preprocessing python=3.11
conda activate mosaic-preprocessing
cd /your/path/to/mosaic-preprocessing
pip install -r requirements.txt
```

Install [GLMsingle](https://github.com/cvnlab/GLMsingle)
```
pip install git+https://github.com/cvnlab/GLMsingle.git
conda list glmsingle #check glmsingle version (1.2 for initial release of MOSAIC)
```

Set up your .env file
```
cp .env.example .env
```
The .env file defines paths to your project directory, datasets, tmp directories for fMRIPrep, etc.

I found it easiest to separate dataset-specific folders (e.g., like what you would download from OpenNeuro) from an aggregated MOSAIC dataset folder like:
```
./datasets/
├── <fMRI DATASET A>/
├── <fMRI DATASET B>/
├── <fMRI DATASET Z>/
├── MOSAIC/
```

The central MOSAIC folder will look like:
```
./datasets/MOSAIC/
├── stimuli/
├── testtrain/
├── hdf5_files/
├── participants/
```

The path to the datasets should be set as an environment variable DATASET_ROOT.
Your path to this repository will likely be in a different directory entirely and set as an environment variable PROJECT_ROOT. 'source' your .env file.

### MOSAIC preprocessing
MOSAIC preprocessing can be divided in two stages: fMRI and stimulus set. Here we describe the preprocessing for the initial set of 8 datasets in the MOSAIC manuscript. If you want to add a ninth dataset that is MOSAIC-compatible with the other eight, for example, follow this pipeline. Otherwise, feel free to preprocess datasets with other pipelines, recognizing that they will not be MOSAIC compatible with the initial eight.

### Stimulus set preprocessing
For each fMRI dataset separately:

1. Download the stimuli from the original fMRI dataset publication. Unfortunately, we do not have permission to redistribute copyrighted material. We provide stimulus download scripts for some of the datasets, but otherwise visit the dataset links above and follow their download instructions.

Input: None

Output: stimulus set downloaded in each dataset-specific folder.

2. Move all stimuli into MOSAIC stimulus folder. Here, the order you move the stimuli into this folder might matter if the stimuli share the same filename.  

1. Extract [DreamSim](https://arxiv.org/abs/2306.09344) embeddings for each stimulus and save. Download the stimuli from the original fMRI dataset publication. MOSAIC does not directly provide the stimuli due to copyright.

INPUTS: folder with stimuli

OUTPUTS: pickle file with dictionary in format {'stimulus_filename': np.array((1768,))}. Keys are stimulus filenames (str), values are DreamSim vector embeddings (ndarray)

```
python src/stimulusSetPreparation/extract_embeddings/dreamsim_embeddings.py
```

2. (Optional) Define dataset-specific train-test splits if not already done. If a dataset already defines a train-test split in its original publication, we highly recommend using preserving this split. If a dataset does not define a train-test split and one is not defined elsewhere, define your own in such a way that each subject has a non-overlapping train-test split. Sometimes this procedure requires additional code with file artifacts (see NOD below) and other times it does not (see HAD).

INPUTS: DreamSim embedding pickle file (or can vary based on how you want to determine test/train split)

OUTPUTS: pickle file containing list of stimuli in test and train splits (but can also vary based on your method).

```
python src/stimulusSetPreparation/extract_dataset_stiminfo/nod_testtrain_splits/make_imagenet_splits_rdm.py
python src/stimulusSetPreparation/extract_dataset_stiminfo/nod_testtrain_splits/make_coco_splits_rdm.py
```

3. Extract detailed stimulus information into a .tsv file. Required columns are
    - filename (original filename of the stimulus)
    - alias (different filename that is dataset-specific)
    - source (original stimulus source, e.g., ImageNet)
    - test_train (whether this stimulus is in the dataset's test or train split)
    - sub-XX reps (how many times sub-XX viewed this stimulus. Each subject is its own column)

INPUTS: fMRI dataset downloaded in BIDS format

OUTPUTS: .tsv file with stimulus information

```
python src/stimulusSetPreparation/extract_dataset_stiminfo/extract_<DATASET>_stiminfo.py
```

MOSAIC emphasizes diligent data provenance. Especially as we train AI models on this data, we want to keep track of exactly which stimuli are used.

### fMRI responses
For each fMRI dataset separately:

1. Download the raw data from the original publication. Organize in [BIDS](https://bids.neuroimaging.io/) format if not done so already.

INPUTS: None

OUTPUTS: Folder with raw fMRI data in BIDS format.

```
bash src/fmriDatasetPreparation/datasets/<DATASET>/download/download_<DATASET>.sh
```

2. Preprocess the data using your pipeline of choice (here, [fMRIPrep](https://fmriprep.org/en/stable/) version 23.2.0). Make sure to keep all arguments the same across datasets, such as registration space, reference slice etc.

INPUTS: Folder with raw fMRI data in BIDS format.

OUTPUTS: fMRIPrep derivatives

```
bash src/fmriDatasetPreparation/datasets/<DATASET>/fmriprep/run_fmriprep_single.sh
```

3. Estimate single trial betas using a General Linear Model (here, using [GLMsingle](https://elifesciences.org/articles/77599) version 1.2). Again, make sure version is consistent across datasets.

INPUTS: fMRIPrep derivatives

OUTPUTS: GLMsingle outputs of single trial beta estimates, pickle file of stimulus order corresponding to the beta estimates

```
python src/fmriDatasetPreparation/datasets/<DATASET>/GLM/glmsingle_<DATASET>.py
```

4. Normalize single-trial beta estimates by dataset-specific train-test splits. This step uses the stimulus information .tsv file from "stimulus set preprocessing" step 3.

INPUTS: GLMsingle outputs of single trial beta estimates, .tsv file from "stimulus set preprocessing" step 3

OUTPUTS: train and test (and artificial) pickle files with normalized beta estimates. Each pickle file has tuple (betas, stimorder)

```
python src/fmriDatasetPreparation/datasets/<DATASET>/GLM/organize_betas_<DATASET>.py
```

5. Compute noise ceiling estimates per voxel using the method detailed in the Natural Scenes Dataset [manuscript](https://www.nature.com/articles/s41593-021-00962-x). 

INPUTS: train and test pickle files with normalized beta estimates. 

OUTPUTS: npy files of noise ceiling estimates per vertex. image of noise ceiling estimates on a flatmap (optional)

Run the notebook located in:
```
src/fmriDatasetPreparation/datasets/<DATASET>/validation/noiseceiling_<DATASET>.ipynb
```

6. Compile fMRI data into one .hdf5 file for each subject individually. Verify the file is MOSAIC compliant (see 'how to make your MOSAIC upload MOSAIC compliant' below).

INPUTS: beta estimate pickle files from step 4, noise ceiling npy files from step 5, subject information, GitHub repository url.

OUTPUTS: .hdf5 file

```
python src/fmriDatasetPreparation/create_hdf5/create_hdf5_pkl.py --subjectID_dataset sub-XX_DATASET --owner_name "firstName lastName" --owner_email youremail@email.com

python src/fmriDatasetPreparation/create_hdf5/create_hdf5_pkl.py --subjectID_dataset sub-01_NSD --owner_name "Benjamin Lahner" --owner_email blahner@mit.edu
```

Note that the .hdf5 files include all single trial beta estimates. Subsequent stimulus set filtering when you aggregate subjects/datasets into your MOSAIC dataset will output train and test set .json files that will simply not reference the stimuli and fMRI trials that get filtered out. But the .hdf5 files themselves are agnostic to this stimulus set filtering.

### Validation
Finally, share some preprocessing validation reports. We recommend sharing fMRIPrep's output (or the equivalent if you are using a different pipeline) and noise ceiling estimates.
- [fMRIPrep reports](https://drive.google.com/drive/folders/1HM_YeygB6IgxbGx_IalKFN66slG4Lxmo?usp=sharing)
- [Noise ceiling estimates](TODO)

### Upload single subject hdf5 files to the MOSAIC website
For a fMRI dataset of n subjects, you will upload n+2 files:

1. n .hdf5 files, one for each of the n subjects.
2. One pickle file containing the DreamSim embeddings for each stimulus. 
3. One .tsv file containing the detailed stimulus set information.

Do not upload the stimuli themselves. Most stimulus sets have copyright restrictions with varying terms and conditions. MOSAIC does not have the rights to redistribute these stimulus sets, so please download them from the original source that should be detailed in the fMRI dataset's original publication. The DreamSim embeddings and .tsv files are enough for MOSAIC to identify train-test splits on the website's backend.

## Merging hdf5 files into a MOSAIC
At this step, you have either preprocessed your own datsets into single subject .hdf5 files, or you have downloaded single subject .hdf5 files 
from the MOSAIC data management portal. While you don't have to merge them, merging them into a single .hdf5 file is helpful for many analyses,
like model training.

I found it helpful to have two copies of MOSAIC data - one for experiments that access individual trials (like model training) and one for experiments
that access data in chunks/batches (like loading all subject data at once). I show how to do both. Either one will work for any case, but depending
on the access patterns you expect to use, one will just be faster. 

Assuming the single subject hdf5 files are in /your/path/to/datasets/MOSAIC/hdf5_files/single_subject

For frequently accessing individual trials:
```
python src/fmriDatasetPreparation/create_hdf5/merge_hdf5_ind.py --input_dir /your/path/to/datasets/MOSAIC/hdf5_files/single_subject ---output_dir /your/path/to/datasets/MOSAIC/hdf5_files/merged --output_file mosaic_ind.hdf5
```

For frequently accessing chunks:
```
python src/fmriDatasetPreparation/create_hdf5/merge_hdf5_chunks.py --input_dir /your/path/to/datasets/MOSAIC/hdf5_files/single_subject ---output_dir /your/path/to/datasets/MOSAIC/hdf5_files/merged --output_file mosaic_chunks.hdf5
```

## Why hdf5 files?
The hdf5 hierarchical [format](https://docs.h5py.org/en/stable/quick.html) is a very handy way of managing large amounts of data. Most useful is the property that you can load a subset of a vector into memory. For example, we store the full 91282 vertices from the whole brain but often times only want to laod and analyze a few hundred or thousand vertices corresponding to an ROI. HDF5 files allow you to load that small subset into memory without loading the full response vector, resulting in significant computational savings.

Additionally, hdf5 files handle concurrent reads nicely for multi-thread processing and allow you to store metadata ('attributes') right next to the data. A single file with
all the data, although large, is much easier to organize and share than hundreds of thousands of individual files.

### How to make a dataset MOSAIC compliant
As explained above, your upload will consist of n+2 files for a fMRI dataset of n subjects.
Each of the n .hdf5 files must follow the format:
```
|- sub-XX_mosaic.hdf5
    |- dataset_name
    |- subjectID
    |- age
    |- sex
    |- visual_angle
    |- owner_name
    |- owner_email
    |- pipeline
    |- trial_format
    |- github_repo_url
    |- publication_url
    |- betas
        |- stimulus_filename_rep-R.npy
        |- ...
    |- noiseceling
        |- noiseceiling_n-N.npy
    |- nan_indices
        |- sub-XX_nan_indices.npy
```
The first entries are type string. The betas and noiseceilings are type float of shape (91282,). The nan_indices are type bool of shape (91282,).

The pickle file with the DreamSim embeddings are a dictionary in the form:
```
{'stimulus-<stimulus_filename>': DreamSim_embedding}
```
where DreamSim_embedding is type float of shape (1768,).

The .tsv file with stimulus information must contain all the columns described above.

## (2) Template

### Loading the fMRI responses
All single-trial fMRI responses output from each individual dataset's GLM is stored in the hdf5 file 'mosaic_version_1_0_0.hdf5'.
Following the nomenclature of the hdf5 hierarchical [format](https://docs.h5py.org/en/stable/quick.html), the file is organized into 93 'groups' representing the 93 subjects across the 8 datasets. The group names follow the pattern 'sub-XX_DATASET'. Each group has 'attributes' of 'age', 'sex', and the 'visual_angle' the stimulus was presented at.
- age: The reported age of the participant in years. Taken from the original dataset's particpants.tsv file.
- sex: The reported sex of the participant, either 'M' for Male or 'F' for Female. Taken from the original dataset's particpants.tsv file.
- visual_angle: The visual angle that the stimulus was presented to the subject, in degrees. A single value means that the stimulus was presented as a square. The visual_angle is the same for all responses within a subject and dataset.

Note that age and sex information for the individual participants in GOD were not available, so their values are n/a.

Within each 'sub-XX_DATASET' group, there exists two sub-groups 'response' and 'noiseceiling' for the single-trial brain responses and noiseceilings. These sub-groups have no attributes. In the 'response' sub-group, each single-trial brain response of shape (91282,1) is input as what the hdf5 format calls a 'dataset'. Each 'dataset' (i.e., response) has attributes 'nan_indices', 'phase', 'repetition', 'presented_stimulus_filename', and 'image_stimulus_filename'. 
- nan_indices: an array of indices that were originally nans. The single-trial brain response has its nan_indices filled in by averaging the values of adjacent vertices (see methods section in the paper for more details). Use this attribute to undo the nan filling or mask the response. Note that all nan_indices across all subjects and datasets are already in a separate file 'nan_indices_dataset.npy' useful to globally mask single-trial responses.
- phase: 'test' or 'train'. This reflects the test/train partition of the orginal dataset (or if not available, what we used as that dataset's test/train partition).
- repetition: integer of the number repetition. This is mainly used just to differentiate repeated stimulus presentations to the same subject and is not meant to carry extra information, like repetition order.
- presented_stimulus_filename: The source filename of the stimulus that was presented to the subject.
- image_stimulus_filename: The source filename of the image version of each stimulus. This is identical to 'presented_stimulus_filename' in all cases except for the short video stimuli of BMD and HAD, where the filename here reflects the middle frame of each video that was often used when image-level information (e.g., embeddings) were needed.

The 'noiseceiling' sub-group contains noise ceilings of shape (91282,1) as 'datasets'. Each 'dataset' (i.e., noise ceiling) has attributes 'nan_indices', 'n', and 'phase'.
- nan_indices: an array of indices that were originally nans from the noiseceiling calculation for that subject, dataset, phase, and n. The noise ceiling response vector has its nan_indices filled in by averaging the values of adjacent vertices (see methods section in the paper for more details). Use this attribute to undo the nan filling or mask the noise ceiling response. Note that all nan_indices across all subjects and datasets are already in a separate file 'nan_indices_dataset.npy' useful to globally mask single-trial responses.
- n: the number of trials the noise ceiling was averaged over. This value is either 1, meaning the noise ceiling is valid for single trial responses, or the number of stimulus repeats, meaning the noise ceiling is valid for trial-averaged responses.
- phase: 'test' or 'train'. This reflects the test/train partition of the orginal dataset (or if not available, what we used as that dataset's test/train partition).

We also provide another HDF5 file, 'mosaic_version-1_0_0_chunks.hdf5', that contains the same data described above but organized differently to allow for faster loading of a subject's entire data. With the chunked version, all responses from a subject can be loaded into memory much more quickly than looping over and loading individual responses in the other hdf5 file. Conversely, random access of responses in the chunked version is much slower.

Example of accessing data in the hdf5 file:
```
with h5py.File(os.path.join(dataset_root,'mosaic_version-1_0_0.hdf5'), 'r') as file:
    print(f"Keys: {file.keys()}") #print the 93 subject keys
    #print the attributes of one example subject
    print(f"Attributes: {file['sub-01_NSD'].attrs.keys()})

    #append slices of the first 10,000 indices of each single-trial brain response of subject 'sub-01_GOD'. Note a big advantage of hdf5 files is that we can partially load the single-trial response.
    data = []
    nan_indices = [] #append nan_indices attribute
    for fname in file['sub-01_GOD']['sub-01_GOD_response'].keys():
        data.append(file['sub-01_GOD']['sub-01_GOD_response'][fname][:10000])
        attributes.append(file['sub-01_GOD']['sub-01_GOD_response'][fname].attrs.get('nan_indices'))
```

This hdf5 file contains all single trial responses aggregated across the 8 datasets. Subsequent cleaning steps, such as identifying stimulus overlap and perceptual similarity, are performed to curate custom train/test json files, not modify the raw data here. These custom train/test splits will just selectively index this hdf5 file.

### Running fMRIPrep
If you want to run the fMRIPrep scripts, follow [fMRIPrep's installation guide](https://fmriprep.org/en/stable/installation.html). The provided scripts run fMRIPrep in a docker container. Note that the environment variable FREESURFER_HOME is used in the fMRIPrep scripts and can be set in either this project's .env file or in your shell RC file (e.g., ~/.bashrc, ~/.zshrc) since it may apply to more than just this project.

### Using Human Connectome Workbench and HCP-utils
[Human Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench) and [HCP-utils](https://rmldj.github.io/hcp-utils/) provide useful functions for preprocessing, visualization, and analysis, especially in the fsLR32k space. Follow their installation instructions linked above.

### fMRI Dataset Preprocesing
All datasets are downloaded after dcm2nii conversion performed by the datasets' authors. We then use fMRIprep version 23.2.0 to preprocess the data. We use GLMsingle (Prince et al., 2022) to estimate single-trial beta values. 

TODO: Each dataset undergoes quality checks using MRIQC (to measure quality at various preprocessing stages but before GLM) and various noiseceiling measures and sanity checks after GLM.

Note that this shared preprocessing pipeline between datasets is intended to be transparent and flexible enough to facilitate the addition of future fMRI datasets. Custom preprocessing pipelines might optimize dataset quality on a per-dataset basis (e.g., the Natural Scenes Dataset with 7T). So, if a researcher wants to only use a single fMRI dataset for a study, they may be better off using a custom and dataset-optimized pipeline.

### Adding a Dataset to MOSAIC
MOSAIC is intended to be an extensible framework where datasets can be added post-hoc. New datasets should be pre-processed in the same way as every other dataset in the group. Compiling datasets is agnostic to the order you add them in all steps (e.g., first adding dataset A, then dataset B, then dataset C will result in the same meta-dataset as first adding dataset B, then C, then A) EXCEPT the step moving the stimuli to a common folder. In this step, a stimulus with the same filename will overwrite an earlier stimulus. This can lead to inconsistencies if a stimulus of the same filename is a cropped version of another stimulus of the same filename in another dataset. If the stimuli are exactly the same between datasets, overwriting will not make a difference.

1. Preprocess with identical fMRIPrep version and parameters as the other datasets
2. Create a DATASET_stiminfo.tsv file that details each stimulus's filename, source, aliases, and which subjects in the dataset saw the stimulus.
3. Estimate single trial beta values using GLMsingle type D betas when repeats are available.
4. Extract DreamSim features from each stimulus
5. Put all stimuli into shared folder
6. Put all single trial betas responses into shared folder
7. Define test/train sets by resolving test/train conflicts and removing perceptually similar training images.
8. Compute subject noise ceilings
9. Fill in nans for single trial betas
10. Fill in nans for noise ceilings
11. Add single trial betas as a 'group' to the hdf5 file

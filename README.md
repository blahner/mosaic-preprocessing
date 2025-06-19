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

If you want to run the fMRIPrep scripts, follow [fMRIPrep's installation guide](https://fmriprep.org/en/stable/installation.html). The provided scripts run fMRIPrep in a docker container. Note that the environment variable FREESURFER_HOME is used in the fMRIPrep scripts and can be set in either this project's .env file or in your shell RC file (e.g., ~/.bashrc, ~/.zshrc) since it may apply to more than just this project.

[Human Connectome Workbench](https://www.humanconnectome.org/software/connectome-workbench) and [HCP-utils](https://rmldj.github.io/hcp-utils/) provide useful functions for preprocessing, visualization, and analysis, especially in the fsLR32k space. Follow their installation instructions linked above.

## Using the MOSAIC dataset
You can use the MOSAIC dataset as-is from the MOSAIC [manuscript](TODO) or you can preprocess your own dataset to add to it.

### I want to use the originally published MOSAIC dataset
To use the version of MOSAIC with 8 datasets published [here](TODO), you need to:
1. Download the subject-specific fMRI data from the [MOSAIC portal](mosaic.csail.mit.edu)
2. Download the train-test json splits from the [MOSAIC portal](mosaic.csail.mit.edu)
3. Download the stimulus sets from their original source (see detailed instructions below).

a. Since it is common for stimulus sets to not be under a Creative Commons license, we do not host or distribute any stimulus sets. Please download the stimulus sets following the instructions of the original publication. We provide download scripts when possible or else point you to the download instructions below:

- BOLD5000: https://bold5000-dataset.github.io/website/download.html
- BOLD Moments Dataset (BMD): https://github.com/blahner/BOLDMomentsDataset
- Generic Object Decoding (GOD) and Deeprecon (same stimulus set): https://github.com/KamitaniLab/GenericObjectDecoding and https://github.com/KamitaniLab/DeepImageReconstruction
- Human Actions Dataset (HAD): https://openneuro.org/datasets/ds004488 and/or the script "src/fmriDatasetPreparation/datasets/HumanActionsDataset/download/ds004488-1.1.1_noderivatives.sh"
- THINGS: Download all THINGS database images https://osf.io/jum2f/. Then this notebook copies the ones used in the fMRI study to other folder: "src/fmriDatasetPreparation/datasets/THINGS_fmri/download/identify_experimental_stimuli.ipynb". Alternatively, we provide a txt file of the list of 8740 stimuli "THINGS_fmri_filenames.txt" in case you don't want to download THINGS event files etc.
- Natural Object Dataset (NOD): https://openneuro.org/datasets/ds004496 and/or the script "src/fmriDatasetPreparation/datasets/NaturalObjectDataset/download/download_nod_stimuli.sh"
- Natural Scenes Dataset (NSD): https://natural-scenes-dataset.s3.amazonaws.com/index.html#nsddata_stimuli/stimuli/ 

Input: None

Output: stimulus set downloaded in each dataset-specific folder.

b. Next, do some light preprocessing to extract the video frames from BMD and HAD, save nsd synthetic stimuli in own folders.
```
python src/stimulusSetPreparation/video_frame_extraction/extract_frames_bmd.py
python src/stimulusSetPreparation/video_frame_extraction/extract_frames_had.py
src/stimulusSetPreparation/compile_datasets/save_nsdsynthetic_stimuli.ipynb
```

Input: mp4 videos and nsd synthetic stimuli .hdf5 file in dataset-specific folder

Output: jpg video frames and jpg nsd synthetic stimuli in dataset-specific folder

c. Next, move all stimuli into MOSAIC stimulus folder. Here, the order you move the stimuli into this folder might matter if the stimuli share the same filename.  
```
bash src/stimulusSetPreparation/compile_datasets/copy_dataset_stim.sh
```

Input: stimuli in their dataset specific folder

Output: stimuli in the MOSAIC stimuli folder

### I want to add my own fMRI dataset to MOSAIC
To add your own fMRI dataset to MOSAIC, you must preprocess it in a specific format in order to make it compatible with the other datasets in MOSAIC. We demonstrate this preprocessing here using the original 8 MOSAIC datasets as an example. MOSAIC preprocessing can be divided in two stages: fMRI and stimulus set. Here we describe the preprocessing for the initial set of 8 datasets in the MOSAIC manuscript. If you want to add a ninth dataset that is MOSAIC-compatible with the other eight, for example, follow this pipeline. Otherwise, feel free to preprocess datasets with other pipelines, recognizing that they will not be MOSAIC compatible with the initial eight.

### Stimulus set preprocessing
1. Follow the steps above to download the stimulus sets and move them into a shared MOSAIC stimuli folder.

INPUTS: None

OUTPUTS: All stimuli are in a shared stimuli folder in your MOSAIC dataset directory

2. Extract [DreamSim](https://arxiv.org/abs/2306.09344) embeddings for each stimulus and save. Download the stimuli from the original fMRI dataset publication. MOSAIC does not directly provide the stimuli due to copyright.

INPUTS: folder with stimuli in your MOSAIC dataset directory

OUTPUTS: folder with dreamsim embeddings as .npy files for each stimulus

```
python src/stimulusSetPreparation/extract_embeddings/dreamsim_embeddings.py
```

3. (Optional) Define dataset-specific train-test splits if not already done. If a dataset already defines a train-test split in its original publication, we highly recommend using preserving this split. If a dataset does not define a train-test split and one is not defined elsewhere, define your own in such a way that each subject has a non-overlapping train-test split. Sometimes this procedure requires additional code with file artifacts (see NOD below) and other times it does not (see HAD).

INPUTS: DreamSim embedding pickle file (or can vary based on how you want to determine test/train split)

OUTPUTS: pickle file containing list of stimuli in test and train splits (but can also vary based on your method).

As an example, neither HAD nor NOD defined test train splits. HAD was simple enough to define based on the experimental runs. NOD was a bit trickier, and our method required these scripts that used dreamsim embeddings.
```
python src/stimulusSetPreparation/extract_dataset_stiminfo/nod_testtrain_splits/make_imagenet_splits_rdm.py
python src/stimulusSetPreparation/extract_dataset_stiminfo/nod_testtrain_splits/make_coco_splits_rdm.py
```

4. Extract detailed stimulus information into a .tsv file. Required columns are
    - filename (original filename of the stimulus)
    - alias (different filename that is dataset-specific)
    - source (original stimulus source, e.g., ImageNet)
    - test_train (whether this stimulus is in the dataset's test or train split)
    - sub-XX reps (how many times sub-XX viewed this stimulus. Each subject is its own column)

INPUTS: events files downloaded in BIDS format (e.g., what you would download from OpenNeuro)

OUTPUTS: .tsv file with stimulus information

```
python src/stimulusSetPreparation/extract_dataset_stiminfo/extract_<DATASET>_stiminfo.py
```

MOSAIC emphasizes diligent data provenance. Especially as we train AI models on this data, we want to keep track of exactly which stimuli are used.

5. Define train-test splits. Each individual subject's test-train splits should not be used in MOSAIC because, when aggregating across datasets, it is common for the same stimulus or a highly similar stimulus (e.g., a crop) to be in the train set of one dataset and test set of another. We want independent test-train splits. Thus, curating this test-train split heavily depends on which datasets are being used in MOSAIC. To preserve the original test-train splits for legacy comparisons, if a stimulus is defined as "test" in dataset A, it will not flip to "train" when curating the aggregated split (and vice versa).

INPUTS: Dataset specific .tsv files from step 4.

OUTPUTS: train.json, test.json, and artificial.json files for the aggregated dataset MOSAIC collection

```
src/stimulusSetPreparation/compile_datasets/compile_stiminfo_acrossdatasets.ipynb #this notebook creates a 'merged_stiminfo.tsv' file
python src/stimulusSetPreparation/compile_datasets/make_testtrain_splits.py 
src/stimulusSetPreparation/compile_datasets/testtrain_stats.ipynb #this notebook gives some stats about the composition of your test-train splits
```

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
For a fMRI dataset of n subjects, you will upload n+1 files:

1. n .hdf5 files, one for each of the n subjects.
2. One .tsv file containing the detailed stimulus set information.

Do not upload the stimuli themselves. Most stimulus sets have copyright restrictions with varying terms and conditions. MOSAIC does not have the rights to redistribute these stimulus sets, so please download them from the original source that should be detailed in the fMRI dataset's original publication.

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

We provide a notebook 'src/fmriDatasetPreparation/create_hdf5/load_hdf5.ipynb' that shows you basic hdf5 loading commands.

## Citation
If you use MOSAIC, please cite the original MOSAIC manuscript and the orginal publications of each of the datasets.
# mosaic-preprocessing
This repository demonstrates how to preprocess a raw BIDS-compatible fMRI dataset (e.g., OpenNeuro) into MOSAIC-compatible fMRI-stimulus pairs. Data are preprocessed into one hdf5 file per subject. All datasets that get added to MOSAIC must be accompanied by a public GitHub repository that details its preprocessing, like this repository. This repository details the preprocessing for the following datasets:

- [BOLD5000](https://openneuro.org/datasets/ds001499)
- [BOLD Moments Dataset (BMD)](https://openneuro.org/datasets/ds005165)
- [Generic Object Decoding (GOD)](https://openneuro.org/datasets/ds001246)
- [Deeprecon](https://openneuro.org/datasets/ds001506)
- [Human Actions Dataset (HAD)](https://openneuro.org/datasets/ds004488)
- [Human Connectome Project (HCP)](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release)
- [THINGS](https://openneuro.org/datasets/ds004192)
- [Natural Object Dataset (NOD)](https://openneuro.org/datasets/ds004496)
- [Natural Scenes Dataset (NSD)](https://registry.opendata.aws/nsd/)

### Create environment
```
conda create -n MOSAIC python=3.11
conda activate MOSAIC
cd /your/path/to/MOSAIC
pip install -r requirements.txt
```

Set up your .env file
```
cp .env.example .env
```

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

### Running MRIQC
If you want to run MRIQC, follow [MRIQC's installation guide](https://mriqc.readthedocs.io/en/latest/install.html). The provided scripts run MRIQC in a docker container.

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
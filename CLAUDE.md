# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Preprocessing pipeline for MOSAIC, an aggregated fMRI dataset spanning 8 source datasets (BOLD5000, BOLD Moments Dataset (BMD), Generic Object Decoding (GOD), Deeprecon, Human Actions Dataset (HAD), THINGS, Natural Object Dataset (NOD), Natural Scenes Dataset (NSD)). The repo serves two purposes: reproducing the exact preprocessing used in the MOSAIC manuscript, and acting as a template for adding a 9th dataset. Read `README.md` for the full step-by-step pipeline narrative — this file only covers what's needed to navigate and modify the code.

## Environment setup

```bash
conda create -n mosaic-preprocessing python=3.11
conda activate mosaic-preprocessing
pip install -r requirements.txt
pip install git+https://github.com/cvnlab/GLMsingle.git   # not in requirements.txt
cp .env.example .env   # then fill in and `source .env`
```

Required env vars (`.env`, loaded via `python-dotenv` in `src/utils/*.py`):
- `PROJECT_ROOT` — path to this repo
- `DATASETS_ROOT` — parent dir holding one folder per raw fMRI dataset, plus `MOSAIC/` for aggregated outputs
- `PYTHONPATH` — should include `PROJECT_ROOT` (scripts do `from src.utils...` imports)
- `TMP` — scratch dir for fMRIPrep intermediates
- `CUDA_VISIBLE_DEVICES` — GPU visibility for embedding extraction / model steps

fMRIPrep runs via Docker — see `fmriDatasetPreparation/datasets/<DATASET>/fmriprep/run_fmriprep_single.sh`. `FREESURFER_HOME` must be set (either in `.env` or shell rc). No test suite exists in this repo.

All non-Docker pipeline scripts (GLMsingle, organize_betas, noise ceiling, QA scripts) must run in the `mosaic-preprocessing` conda env — it's the only env with `GLMsingle`/`hcp_utils`/`nilearn` installed alongside the `requirements.txt`-pinned `numpy==1.26.4`. On at least this machine, bare `python3` and even `conda run -n mosaic-preprocessing python3` silently resolve to a *different* conda env because another env's `bin/` is prepended to `$PATH` ahead of the base conda install (not from this repo's `.bashrc` block — some other shell init). Don't trust `conda activate`/`conda run` here without verifying `python3 -c "import sys; print(sys.prefix)"` actually prints `.../envs/mosaic-preprocessing`; when in doubt, invoke the env's binary by its full path, e.g. `/data/vision/oliva/blahner/anaconda3/envs/mosaic-preprocessing/bin/python3`.

## Architecture

### Two parallel top-level pipelines under `src/`

**`fmriDatasetPreparation/`** — per-subject fMRI processing. `datasets/<DATASET>/` (one dir per source dataset: `BOLD5000`, `BOLDMomentsDataset`, `deeprecon`, `GenericObjectDecoding`, `HumanActionsDataset`, `NaturalObjectDataset`, `NaturalScenesDataset`, `THINGS_fmri`) each follow the same subfolder pattern and pipeline stage order:
```
download/          → raw BIDS data from source (most datasets; BMD has none, downloaded via download_stimuli.sh)
fmriprep/           → run_fmriprep_single.sh (must keep args e.g. output space identical across datasets)
GLM/                → glmsingle_<dataset>.py (single-trial betas) then organize_betas_<dataset>.py (normalize by train/test split)
temporal_filtering/ → dataset-specific temporal filtering of timeseries data
validation/         → noise ceiling notebooks, QC notebooks specific to that dataset
```
Dataset-specific scripts are named with a lowercase dataset abbreviation suffix (bmd, had, nod, etc.) matching the abbreviations used in `subjectID_dataset` identifiers (see below).

Cross-dataset steps live directly under `fmriDatasetPreparation/`:
- `create_hdf5/create_hdf5_pkl.py` — compiles one subject's betas + noise ceiling into a MOSAIC-compliant `.hdf5` (invoked with `--subjectID_dataset sub-XX_DATASET`)
- `create_hdf5/merge_hdf5_ind.py` vs `merge_hdf5_chunks.py` — merge single-subject hdf5s into one aggregate file, optimized for individual-trial access vs batch/chunk access respectively (pick based on downstream access pattern, e.g. model training vs bulk loading)
- `visualizations/*_fmriprep_qc_movie.py` — one QC-movie script per dataset (see below)
- `qa/` — cross-dataset QA tooling, e.g. `coreg_determinant_check.py` (flags fMRIPrep runs whose BOLD→T1w coregistration determinant is a robust per-subject outlier — every dataset's fmriprep script passes the non-default `--bold2t1w-dof 12`, which occasionally lets a run's registration collapse to a spurious affine scale instead of pure rigid motion). See `qa/README.md` for the single-subject reprocessing pattern (new `derivatives/versionX`, never overwrite an existing version) used to test a preprocessing fix on one subject without touching the rest of a dataset, and for the batch/background multi-subject variant (`datasets/BOLDMomentsDataset/fmriprep/reprocess_all_bmd_subjects.sh`) — check `qa/README.md` section 3 first if you see a `derivatives/versionD` or `reprocess_logs/` for BMD and need to know what's running and why.

**`stimulusSetPreparation/`** — stimulus-side pipeline, independent of the fMRI pipeline until the final compile step:
1. `download_stimuli.sh` — per-dataset download/instructions (most stimulus sets aren't redistributable — this repo never hosts stimuli itself)
2. `video_frame_extraction/` — BMD/HAD are video datasets; extract frames before embedding
3. `extract_embeddings/dreamsim_embeddings.py` — DreamSim embeddings per stimulus
4. `extract_dataset_stiminfo/extract_<dataset>_stiminfo.py` — per-dataset `.tsv` with filename/alias/source/test_train/subject-repetition-count columns
5. `compile_datasets/` — merges per-dataset stiminfo into one MOSAIC-wide train/test/artificial split. Split logic is "sticky": once a stimulus is `test` in any one dataset's original split, it stays `test` in the aggregate even if another dataset would've put it in `train` — this avoids the same/near-duplicate image appearing in train for one dataset and test for another. Threshold for this is `OUTLIER_CUTOFF` in `make_testtrain_splits.py`.

### Cross-dataset conventions

- **Subject identifiers** are always `sub-XX_DATASET` (e.g. `sub-01_BMD`, `sub-05_NOD`) — this is the join key between hdf5 filenames, `src/utils/dataset.py`'s `FMRIDataset`, and `src/utils/helpers.py`'s `FilterDataset`. `FilterDataset.datasets_subjects` in `helpers.py` is the canonical list of subject counts per dataset — check there before assuming a subject range.
- **Directory layout on disk**: `$DATASETS_ROOT/<RawDatasetName>/` (BIDS + fMRIPrep derivatives, per-dataset naming e.g. `BOLDMomentsDataset`) is kept separate from `$DATASETS_ROOT/MOSAIC/{stimuli,testtrain,hdf5_files,participants}/` (the aggregated output). Don't assume dataset-specific derivatives live under `MOSAIC/`.
- **HDF5 as interchange format**: chosen for partial-vector reads (e.g. loading a handful of ROI vertices out of the full 91282-vertex whole-brain vector without reading the rest) and safe concurrent multi-thread reads. Single-subject hdf5s are the unit of distribution; merging is a separate, optional step for convenience.
- **Not all subjects have a noise ceiling**: HAD (no repeats for any subject) and NOD subjects 10-30 (no repeats) lack noise ceiling data — `FilterDataset.subjects_nonoiseceiling` in `helpers.py` encodes this.

### QC movie scripts (`fmriDatasetPreparation/visualizations/*_fmriprep_qc_movie.py`)

One script per dataset, each producing a single MP4 walking through every subject with: per-session tSNR maps, per-run FD/DVARS motion summary (one dot per run, colored by session), BOLD↔T1w coregistration quality, cortical surface/ribbon QC, and an animated BOLD clip covering every run of every session. All 8 scripts share near-identical structure but are NOT abstracted into a shared module — when fixing a bug or changing behavior in one, check whether the same code exists (often byte-identical) in the others and apply consistently, since they were deliberately kept as parallel copies given how much dataset-specific session/task naming varies (e.g. deeprecon's semantic non-numeric session names vs NSD's `ses-nsd01..40` vs BMD's simple `ses-02..05`). `bmd_fmriprep_qc_movie.py` is the one exception with a materially different (non-dynamic) session-discovery implementation.

All 7 non-BMD scripts discover sessions/tasks/runs dynamically by globbing confound-timeseries filenames (`list_functional_sessions`, `list_runs` helpers) rather than assuming a fixed template, since run counts and session-naming conventions vary per dataset and even per subject within a dataset (e.g. BOLD5000's anat-only session number differs between sub-CSI1-3 and sub-CSI4).

# QA / single-subject reprocessing scaffolding

Tooling for finding and fixing preprocessing problems in an individual
subject without touching the rest of a dataset's derivatives. Grew out of
diagnosing a brain-scaling artifact in BOLDMomentsDataset sub-05 (see git
history / PR description for the full writeup); kept general so it applies
to any dataset here, current or future.

## 1. `coreg_determinant_check.py` — find suspect runs automatically

All 9 datasets' fmriprep scripts pass `--bold2t1w-dof 12` (full affine BOLD→T1w
registration) instead of fMRIPrep's own default of 6 (rigid body). Since a
subject's brain cannot actually change size or shear between a functional run
and its own anatomical, any non-unit scale in that per-run coregistration
transform is a registration artifact. Under dof=12 this occasionally happens
on individual runs, unpredictably — it does **not** mean every run, subject,
or dataset using dof=12 is affected, only that some runs might be, silently
(fMRIPrep doesn't warn or fail when it happens).

This script scans every run's `*_from-boldref_to-T1w_..._coreg_xfm.txt`
under a subject's fmriprep output (fMRIPrep's own standard filename, so this
works unmodified on any dataset/version) and flags runs whose coregistration
determinant is a robust outlier relative to that subject's own other runs.

```bash
python src/fmriDatasetPreparation/qa/coreg_determinant_check.py \
  --fmriprep-dir $DATASETS_ROOT/<Dataset>/derivatives/<version>/fmriprep \
  [--subs sub-05 sub-07] [--mad-thresh 3.5] [--out-csv report.csv]
```

Run this on any dataset before/after a preprocessing change as a fast,
visual-QC-movie-free sanity check. A flagged run is a strong prior that its
functional data has a real spatial normalization error — worth spot-checking
in the QC movie, not an automatic verdict.

## 2. Single-subject reprocessing pattern (never overwrite existing derivatives)

To test a preprocessing fix on one subject without disturbing existing
results for the rest of the dataset:

1. **New derivatives version, not in place.** Copy the dataset's
   `fmriprep/run_fmriprep_single*.sh`, restrict the subject loop to the one
   subject being tested, point `OUTPUT_RELPATH` at a new sibling version
   directory (e.g. `versionD` next to `versionC`), and change only the
   parameter under test. Use a dedicated `WORK` dir too so it can't collide
   with a concurrent/previous run. Pre-create the docker bind-mount output
   directory yourself (`mkdir -p .../fmriprep`) before `docker run` — on this
   filesystem, letting Docker auto-create the bind-mount source can hit an
   NFS permission error.
   Worked example: see `datasets/BOLDMomentsDataset/fmriprep/` — the sub-05
   `--bold2t1w-dof 6` test used this exact pattern against `versionD`.
2. **Downstream scripts take `--version` (default `versionC`).**
   `datasets/BOLDMomentsDataset/GLM/glmsingle_bmd.py`,
   `organize_betas.py`, and `visualizations/bmd_fmriprep_qc_movie.py` all
   accept `--version <name>` to read/write a non-default derivatives version,
   with `versionC` as the default so every existing invocation is unaffected.
   Apply the same one-line parameterization to other datasets' GLM/QC
   scripts before running a similar single-subject test on them.
3. **Noise ceiling: use `noiseceiling_compare.py`, not the notebook.**
   `noiseceiling_bmd.ipynb` loops over all 10 subjects and writes directly
   into the shared `$DATASETS_ROOT/MOSAIC/noiseceilings/` aggregate — running
   it unmodified during a single-subject test would overwrite production
   noise-ceiling files for every subject. `validation/noiseceiling_compare.py`
   computes noise ceiling for one subject against one derivatives version and
   writes into that version's own `GLM/<subject>/noiseceiling/` folder
   instead, so before/after arrays can be diffed directly without touching
   `MOSAIC/noiseceilings/`.
4. **QC movie**: `bmd_fmriprep_qc_movie.py --subs sub-XX --version versionD`
   writes to a version-tagged filename (`..._versionD_qc.mp4`) so it can't
   clobber the dataset-wide `..._all_qc.mp4`.

None of this touches `versionC` (or any dataset's primary derivatives) at any
point — it's an entirely parallel tree, disposable if the test doesn't pan
out.

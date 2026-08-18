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

## 2. `reprocess_bmd_subject.sh` — full pipeline, one launch, never overwrites

Runs fMRIPrep → coreg-determinant QA gate → GLMsingle (all 4 sessions) →
organize_betas → noiseceiling_compare → QC movie for **one subject** as a
single blocking script:

```bash
src/fmriDatasetPreparation/qa/reprocess_bmd_subject.sh <subject_num e.g. 05> <new_version e.g. versionD> [extra fmriprep args...]
# e.g. the sub-05 dof=6 test:
src/fmriDatasetPreparation/qa/reprocess_bmd_subject.sh 05 versionD --bold2t1w-dof 6
```

Because `docker run` blocks until fMRIPrep finishes and every downstream step
is a plain CLI call, this whole chain can be launched **once** in the
background and produces a single completion notification at the end —
important for agentic use: chaining separate backgrounded stages by hand
(fmriprep, *then* notice it finished, *then* launch GLMsingle, ...) across
turns is fragile — a stage can finish while something else is being handled
and its completion notification gets missed, silently stalling the pipeline
for hours until someone asks "is this still running?". One script, one
launch, no stage to forget.

It refuses to run if the target `derivatives/<version>` already exists (so
it can never overwrite `versionC` or any prior test version), and the coreg
QA gate does not halt the pipeline on flagged runs (`set -e` would abort the
whole script) — it writes a report and continues, since a flagged run is a
strong prior worth a human look, not an automatic verdict.

Adapting this to another dataset: copy the script, swap in that dataset's
`fmriprep/run_*.sh` docker invocation and GLM/validation script paths. The
three things every dataset's copy needs are the same ones already applied to
BMD's scripts:
- A `--version` flag (default the dataset's current primary version, e.g.
  `versionC`) on its `glmsingle_*.py`, `organize_betas*.py`, and
  `*_fmriprep_qc_movie.py`, so every existing invocation stays unaffected.
- A single-subject noise-ceiling script instead of the dataset's notebook —
  check first whether that dataset's noise-ceiling notebook writes into the
  shared `$DATASETS_ROOT/MOSAIC/noiseceilings/` aggregate for all subjects
  (BMD's does); if so, running it unmodified during a single-subject test
  would overwrite every other subject's production noise-ceiling file.
- Pre-create the docker bind-mount output dir (`mkdir -p .../fmriprep`)
  before `docker run` — on this filesystem, letting Docker auto-create it
  hits an NFS permission error.

None of this touches a dataset's primary derivatives version at any point —
it's an entirely parallel tree, disposable if the test doesn't pan out.

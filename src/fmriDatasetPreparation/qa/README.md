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

## 2. `datasets/BOLDMomentsDataset/fmriprep/reprocess_bmd_subject.sh` — full pipeline, one launch, resumable

Lives alongside `run_fmriprep_single.sh` (not in this `qa/` folder) since,
unlike `coreg_determinant_check.py`, it's not dataset-agnostic — it directly
encodes BMD's docker invocation and BMD's GLM/validation script paths.

Runs fMRIPrep → coreg-determinant QA gate → GLMsingle (all 4 sessions) →
organize_betas → noiseceiling_compare → QC movie for **one subject** as a
single blocking script:

```bash
src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/fmriprep/reprocess_bmd_subject.sh <subject_num e.g. 05> <new_version e.g. versionD> [extra fmriprep args...]
# e.g. the sub-05 dof=6 test:
src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/fmriprep/reprocess_bmd_subject.sh 05 versionD --bold2t1w-dof 6
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

**Resumable.** A multi-hour background job going silent partway through with
no clean completion/failure notification is a real, observed failure mode
here (happened once during the sub-05 test — GLMsingle stopped mid-session
for 32+ hours with nothing in the log to explain why). Re-running this
script does not restart from scratch: fMRIPrep is skipped if
`fmriprep/sub-XX.html` already exists, and each GLMsingle session is skipped
if its `TYPED_FITHRF_GLMDENOISE_RR.npy` already exists. It always refuses to
target `versionC` specifically (the dataset's primary version), regardless
of resume state. The coreg QA gate does not halt the pipeline on flagged
runs (`set -e` would abort the whole script) — it writes a report and
continues, since a flagged run is a strong prior worth a human look, not an
automatic verdict.

`noiseceiling_compare.py` also saves a flat-map PNG (both hemispheres, same
style as `noiseceiling_bmd.ipynb`) next to each `.npy` it writes, for visual
before/after comparison without needing the notebook.

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

## 3. `datasets/BOLDMomentsDataset/fmriprep/reprocess_all_bmd_subjects.sh` — batch, background, multi-day

Runs `reprocess_bmd_subject.sh` for a **list** of subjects, a few at a time,
in the background. Read this section if you land in a fresh session and see
`derivatives/versionD` or a `reprocess_logs/` folder for BMD and need to
reconstruct what's going on — that's what this is.

```bash
src/fmriDatasetPreparation/datasets/BOLDMomentsDataset/fmriprep/reprocess_all_bmd_subjects.sh <new_version> <max_parallel> "<space-separated subject list>" [extra fmriprep args...]
```

**Status as of 2026-08-20** (update this when the situation changes):
running `reprocess_all_bmd_subjects.sh versionD 2 "01 02 05 08 09" --bold2t1w-dof 6`
in the background. Why these 5 specifically, and not all 10:

- Root cause (see section 1): `--bold2t1w-dof 12` occasionally lets a run's
  BOLD→T1w coregistration converge to a spurious affine scale. Running
  `coreg_determinant_check.py` against the *existing* `versionC` fmriprep
  output for **all 10** BMD subjects found flagged runs in only 5 of them:
  sub-01 (1/62), sub-02 (1/62), sub-05 (11/62, the one that started this),
  sub-08 (4/62), sub-09 (7/62). sub-03, 04, 06, 07, 10 showed **zero**
  flagged runs even under dof=12.
- The dataset filesystem was at 96% full (763GB free) when this started, and
  one subject's full versionD output (fmriprep + GLM + QC movie) is ~54GB —
  reprocessing all 10 would cost ~540GB vs. ~270GB for the 5 affected ones,
  a meaningful difference on a shared disk other lab members' jobs also
  depend on. This was a deliberate scope decision to bound disk usage, not a
  claim that dof=6 wouldn't also modestly help the other 5 — if disk frees
  up and it's worth doing later, re-run this script with those subjects
  added to the list (already-completed subjects/stages are skipped, see
  section 2's "Resumable" note — this applies per-subject here too).

**To check progress**, either subject-by-subject:
```bash
tail -f $DATASETS_ROOT/BOLDMomentsDataset/derivatives/versionD/reprocess_logs/sub-01.log
```
or across all of them at once, e.g. which pipeline stage each is currently on:
```bash
for f in $DATASETS_ROOT/BOLDMomentsDataset/derivatives/versionD/reprocess_logs/sub-*.log; do echo "== $f =="; tail -3 "$f"; done
```
`ps aux | grep -iE "glmsingle_bmd|fmriprep"` and `docker ps` show what's
actively running right now.

**If it stops** (this has happened before with no explanation in the log —
see section 2's "Resumable" note): just re-run the same command. Every
subject and every stage within `reprocess_bmd_subject.sh` checks for its own
completion marker first, so this resumes rather than restarts. It does not
auto-retry within a single invocation (see the script's own comments); if a
subject's log doesn't end with `### DONE:`, re-running the batch script will
retry that subject correctly.

**Concurrency (`max_parallel=2` above)**: each `reprocess_bmd_subject.sh`
invocation runs one Docker fMRIPrep container requesting 16 CPUs, so 2
concurrent ≈ 32 of this machine's 72 cores — left headroom since this is a
shared machine (other users' processes were also active when this started).
Raise it if the machine is otherwise idle, lower it if others are running
heavy jobs.

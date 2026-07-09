#!/usr/bin/env python3
"""
fMRIPrep quality-assurance movie for deeprecon.

Produces a single MP4 that walks through every subject.  For each subject the
movie shows:

  1. Temporal stability
       - tSNR map (mean / std across all volumes, all runs of one session)
       - Framewise displacement across all runs of that session
       - Per-run mean FD + mean std-DVARS across ALL sessions  ← cross-session stability

  2. BOLD ↔ T1w registration accuracy
       - Reference EPI (BOLD) shown in its native space, matching the fMRIPrep
         "Alignment of functional and anatomical MRI data" report panel
       - T1w white-matter mask resampled into EPI space, red contour overlay
         (using the actual ITK coreg transform, applied with SimpleITK)

  3. Cortical surface quality
       - Dense coronal mosaic (8 slices spanning the full brain) with ribbon
         mask in orange on T1w — gaps in the ribbon where T1w shows gray
         matter = missed gyrus
       - Pial (red) and white-matter (cyan) surface contours overlaid

  4. Animated BOLD time-series (clip covering every run of every session)

deeprecon has ~24 functional sessions per subject across several
perception sub-types (NaturalImageTraining x15, NaturalImageTest x3,
ArtificialImage x2, LetterImage x1) plus imagery sessions, all sharing the
"perception"/"imagery" task labels with semantic (non-numeric) session
names. Sessions/tasks/runs are discovered dynamically by globbing rather
than assumed from a fixed template, and by default only the main
"perception" task is included (use --tasks imagery to add the mental-imagery
runs). All sessions (~24) are included by default; pass --max_sessions to
evenly subsample down to a shorter movie if desired.

Usage
-----
  python deeprecon_fmriprep_qc_movie.py [--tasks perception] [--max_sessions N]
                                         [--subs sub-01 sub-02 ...]

Output
------
  OUT_DIR / deeprecon_<subs>_qc_small.mp4
"""

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import nibabel as nib
import pandas as pd
import SimpleITK as sitk

# ---------------------------------------------------------------------------
DATASET_NAME = "deeprecon"
FMRIPREP = Path(os.getenv("DATASETS_ROOT")) / "deeprecon/derivatives/fmriprep"
OUT_DIR  = Path(os.getenv("PROJECT_ROOT")) / "src/fmriDatasetPreparation/visualizations/output/qc"

DEFAULT_TASKS = ['perception']   # main experimental task; task-imagery is excluded by default

FPS      = 4          # frames per second for the output movie
DPI      = 120       # figure DPI  (figsize=(16,9) → 1920×1080)
FIG_SIZE = (16, 9)

# matplotlib's default figure.dpi (100) overrides FIG_SIZE*DPI on every
# individual plt.figure()/subplots() call unless we set this — without it,
# panels render at 1600x900 regardless of DPI above.
plt.rcParams["figure.dpi"] = DPI

HOLD_TITLE      = FPS * 2   # 2 s
HOLD_PANEL      = FPS * 4   # 4 s per static QC panel
BOLD_TARGET_VOLS = 30       # target frames per run in the BOLD clip (sub-sampled)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def norm01(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)


def contrast_enhance(x, lo_pct=2, hi_pct=98):
    """Percentile-clip contrast stretch, used for the EPI reference image."""
    fg = x[x > 0]
    lo, hi = np.percentile(fg, lo_pct), np.percentile(fg, hi_pct)
    return np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)


def center_slices(vol):
    nz = np.argwhere(vol > 0)
    if len(nz) == 0:
        return tuple(s // 2 for s in vol.shape)
    return tuple(nz.mean(0).astype(int))


def fig_to_array(fig):
    """Render a matplotlib figure to an HxWx3 uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[:, :, :3].copy()


def hold(arr, n_frames):
    return [arr] * n_frames


def load_bold(path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4 and data.shape[3] == 1:
        data = data[..., 0]
    return data, img.affine


# deeprecon keeps anatomy in a dedicated ses-anatomy session (no func/), and
# has ~24 semantically-named functional sessions per subject with run counts
# that vary by session type. Sessions, tasks, and runs are discovered by
# globbing instead of assumed from a fixed template.

def find_anat_dir(sub):
    """Locate the directory containing this subject's anatomical outputs."""
    sub_dir = FMRIPREP / sub
    candidates = sorted(list(sub_dir.glob("anat")) + list(sub_dir.glob("ses-*/anat")))
    if not candidates:
        raise FileNotFoundError(f"no anat/ directory found for {sub}")
    return candidates[0]


def find_anat_file(anat_dir, sub, suffix):
    """
    Glob for `{sub}...{suffix}` inside anat_dir, excluding any standard-space
    (`_space-*`) duplicate so we always get the native T1w-space version.
    """
    matches = sorted(p for p in anat_dir.glob(f"{sub}*{suffix}") if "_space-" not in p.name)
    if not matches:
        raise FileNotFoundError(f"no *{suffix} file found in {anat_dir}")
    return matches[0]


def list_functional_sessions(sub, tasks, max_sessions=None):
    """
    Every session (or the bare subject dir, for datasets with no session
    level) that has a func/ subfolder AND at least one run matching one of
    `tasks` — this naturally excludes anat-only sessions without needing to
    hardcode their name, and avoids wasting max_sessions slots on sessions
    that don't contain any of the requested tasks (e.g. localizer-only
    sessions when the requested task is the main experimental one). Evenly
    subsampled down to max_sessions if given.
    """
    sub_dir = FMRIPREP / sub
    candidates = []
    if (sub_dir / "func").exists():
        candidates.append(None)
    candidates += [d.name for d in sorted(sub_dir.glob("ses-*")) if (d / "func").exists()]

    sessions = []
    for ses in candidates:
        fd = (sub_dir / "func") if ses is None else (sub_dir / ses / "func")
        prefix = sub if ses is None else f"{sub}_{ses}"
        if any(list(fd.glob(f"{prefix}_task-{t}*_desc-confounds_timeseries.tsv")) for t in tasks):
            sessions.append(ses)

    if max_sessions is not None and len(sessions) > max_sessions:
        idx = sorted(set(np.linspace(0, len(sessions) - 1, max_sessions).astype(int)))
        sessions = [sessions[i] for i in idx]
    return sessions


def func_dir(sub, ses):
    return (FMRIPREP / sub / "func") if ses is None else (FMRIPREP / sub / ses / "func")


def sub_ses_prefix(sub, ses):
    return sub if ses is None else f"{sub}_{ses}"


def list_runs(sub, ses, task):
    """
    Discover run identifiers (raw string as it appears in the filename, e.g.
    "01", or None if the scan has no run entity) for `task` in this session,
    based on which confounds TSVs actually exist — robust to whichever
    zero-padding/run-count convention the dataset uses.
    """
    fd = func_dir(sub, ses)
    if not fd.exists():
        return []
    prefix = sub_ses_prefix(sub, ses)
    pattern = re.compile(
        rf"^{re.escape(prefix)}_task-{re.escape(task)}(?:_run-(\d+))?_desc-confounds_timeseries\.tsv$")
    runs = []
    for p in sorted(fd.glob(f"{prefix}_task-{task}*_desc-confounds_timeseries.tsv")):
        m = pattern.match(p.name)
        if m:
            runs.append(m.group(1))
    return runs


def func_stub(sub, ses, task, run):
    prefix = sub_ses_prefix(sub, ses)
    return f"{prefix}_task-{task}" + (f"_run-{run}" if run else "")


# ---------------------------------------------------------------------------
# ITK coregistration transform
# ---------------------------------------------------------------------------

def resample_to_reference(moving_path, reference_path, xfm_path, invert=False,
                           interpolator=sitk.sitkLinear):
    """
    Resample `moving_path` onto the voxel grid of `reference_path` using the
    ITK coreg transform, via SimpleITK.

    fMRIPrep's affine .txt transforms follow the ITK convention: applying the
    transform (as read, no inversion) maps a point in the *fixed*/reference
    image's space to the corresponding point in the *moving* image's space —
    this is what ITK/ANTs resampling always needs internally, regardless of
    what the "from-X_to-Y" filename segment implies semantically. So
    "*_from-boldref_to-T1w_..._xfm.txt", used as-is, maps T1w -> boldref.
    Getting this backwards (or re-deriving the matrix algebra by hand)
    silently produces a plausible-looking but wrong registration. Letting
    SimpleITK apply the transform avoids that entirely.
    """
    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)
    reference = sitk.ReadImage(str(reference_path), sitk.sitkFloat32)
    xfm = sitk.ReadTransform(str(xfm_path))
    if invert:
        xfm = xfm.GetInverse()
    out = sitk.Resample(moving, reference, xfm, interpolator, 0.0)
    # SimpleITK arrays are z,y,x -> transpose to nibabel's x,y,z convention
    return sitk.GetArrayFromImage(out).transpose(2, 1, 0)


def surface_contour_on_slice(coords, aff, axis, slice_idx, ax, color, tol=1.5):
    inv_aff = np.linalg.inv(aff)
    vox = (inv_aff[:3, :3] @ coords.T).T + inv_aff[:3, 3]
    near = np.abs(vox[:, axis] - slice_idx) < tol
    if near.sum() == 0:
        return
    pts = vox[near]
    dims = [d for d in [0, 1, 2] if d != axis]
    ax.scatter(pts[:, dims[0]], pts[:, dims[1]],
               s=0.4, c=color, alpha=0.6, linewidths=0)


# ---------------------------------------------------------------------------
# QC panels → matplotlib Figure objects
# ---------------------------------------------------------------------------

def make_title_fig(sub):
    fig = plt.figure(figsize=FIG_SIZE)
    fig.patch.set_facecolor("black")
    fig.text(0.5, 0.55, f"Subject  {sub}", color="white",
             ha="center", va="center", fontsize=36, fontweight="bold")
    fig.text(0.5, 0.40, f"{DATASET_NAME} — fMRIPrep QC", color="#aaaaaa",
             ha="center", va="center", fontsize=20)
    return fig


def make_tsnr_fig(sub, sessions, tasks):
    """tSNR map for every (session, task) pair (all runs pooled) — one slide each."""
    figs = []
    for ses in sessions:
        for task in tasks:
            runs = list_runs(sub, ses, task)
            if not runs:
                continue
            sum_arr = sum_sq_arr = None
            count = 0
            for run in runs:
                p = func_dir(sub, ses) / (
                    f"{func_stub(sub, ses, task, run)}"
                    "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
                if not p.exists():
                    continue
                data, _ = load_bold(p)
                if sum_arr is None:
                    sum_arr = data.sum(3); sum_sq_arr = (data**2).sum(3)
                else:
                    sum_arr += data.sum(3); sum_sq_arr += (data**2).sum(3)
                count += data.shape[3]
            if sum_arr is None:
                continue
            mean  = sum_arr / count
            var   = np.maximum(sum_sq_arr / count - mean**2, 0)
            tsnr  = np.where(var > 0, mean / np.sqrt(var), 0.0)
            brain = mean > 0.1 * mean.max()
            tsnr[~brain] = 0
            label = ses if ses else "sub"

            _, _, cz = center_slices(brain.astype(float))
            fig, ax = plt.subplots(figsize=FIG_SIZE)
            fig.patch.set_facecolor("#111111")
            im = ax.imshow(tsnr[:, :, cz].T, origin="lower", cmap="hot", vmin=0, vmax=100)
            med = np.median(tsnr[brain])
            fig.suptitle(
                f"{sub}  |  Temporal SNR — {label}/{task}  (all runs pooled)\n"
                f"z={cz}   {count} vols   median tSNR = {med:.1f}",
                color="white", fontsize=13)
            ax.axis("off")
            plt.colorbar(im, ax=ax, label="tSNR (unitless)", fraction=0.046, pad=0.04)
            plt.tight_layout()
            figs.append((f"{label}/{task}", fig))

    if not figs:
        fig, ax = plt.subplots(figsize=FIG_SIZE)
        ax.text(0.5, 0.5, "No BOLD data found", ha="center", transform=ax.transAxes)
        figs.append((None, fig))

    return figs


def make_run_summary_fig(sub, sessions, tasks):
    """Per-run mean FD and mean std-DVARS across ALL sessions — one dot per run."""
    run_labels, mean_fd, mean_dvars, ses_ids = [], [], [], []
    session_boundaries = []

    for ses in sessions:
        first = True
        for task in tasks:
            for run in list_runs(sub, ses, task):
                p = func_dir(sub, ses) / (
                    f"{func_stub(sub, ses, task, run)}_desc-confounds_timeseries.tsv")
                if not p.exists():
                    continue
                df = pd.read_csv(p, sep="\t")
                fd_vals = df["framewise_displacement"].dropna().values.astype(float)
                dv_vals = df["std_dvars"].dropna().values.astype(float)
                if first:
                    session_boundaries.append(len(run_labels))
                    first = False
                label = ses if ses else "sub"
                run_labels.append(f"{label}\n{task}" + (f"-{run}" if run else ""))
                mean_fd.append(fd_vals.mean())
                mean_dvars.append(dv_vals.mean() if len(dv_vals) else np.nan)
                ses_ids.append(sessions.index(ses))

    x = np.arange(len(run_labels))
    cmap = plt.cm.get_cmap("tab20", len(sessions))
    colors = [cmap(s) for s in ses_ids]

    fig, axes = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    fig.patch.set_facecolor("#111111")
    for ax in axes:
        ax.set_facecolor("#111111")

    axes[0].scatter(x, mean_fd,    c=colors, s=40, zorder=3)
    axes[0].plot(x, mean_fd,       color="gray", lw=0.8, zorder=2)
    axes[0].set_ylabel("Mean FD per run (mm)", color="white")

    axes[1].scatter(x, mean_dvars, c=colors, s=40, zorder=3)
    axes[1].plot(x, mean_dvars,    color="gray", lw=0.8, zorder=2)
    axes[1].set_ylabel("Mean std-DVARS per run", color="white")

    for xi in session_boundaries[1:]:
        for ax in axes:
            ax.axvline(xi - 0.5, color="white", lw=1, ls="--", alpha=0.4)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=6, color="white", rotation=45, ha="right")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("white")

    # Figure-level legend, with column count adapted to label length so it
    # stays within the canvas width for any session-name style (deeprecon's
    # long "perceptionNaturalImageTraining13"-style names vs. other
    # datasets' short "nsd01" style), and stacked in rows (reserved via
    # measured height) instead of a single axes-anchored row — a wide
    # single-row legend previously forced tight_layout to squeeze the plot
    # axes down to a sliver to keep the legend on-canvas (this is exactly
    # what was making the FD/DVARS plots look "jammed" to one side).
    labels = [(ses or "sub").removeprefix("ses-") for ses in sessions]
    avg_len = sum(len(l) for l in labels) / len(labels)
    row_budget = 150  # empirical char-budget that fits FIG_SIZE width at fontsize~6.5
    ncol = max(1, min(len(sessions), round(row_budget / (avg_len + 3))))
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=cmap(i), markersize=5, label=labels[i])
               for i in range(len(sessions))]
    legend = fig.legend(handles=handles, facecolor="#333333", labelcolor="white",
                         fontsize=6.5, ncol=ncol, loc="upper center",
                         bbox_to_anchor=(0.5, 1.0), frameon=True, columnspacing=1.0,
                         handletextpad=0.4)

    # Measure the legend's actual rendered height (in figure-fraction
    # coords) instead of guessing a per-row margin, so the title/axes
    # placement below it stays correct for any session count.
    fig.canvas.draw()
    legend_bottom_fig = legend.get_window_extent(fig.canvas.get_renderer()) \
        .transformed(fig.transFigure.inverted()).y0
    title_y = legend_bottom_fig - 0.02
    fig.suptitle(
        f"{sub}  |  Per-run motion & signal quality across all sessions\n"
        "flat lines = consistent preprocessing; dots coloured by session",
        color="white", fontsize=13, y=title_y, va="top")

    plt.tight_layout(rect=[0, 0, 1, title_y - 0.06])
    return fig


def make_registration_fig(sub, sessions, tasks, wm_path):
    """
    BOLD → T1w coregistration quality: one slide per session (using the first
    discovered task/run in that session).

    Mirrors fMRIPrep's own "Alignment of functional and anatomical MRI data"
    reportlet: the reference EPI (boldref) is shown in its native space,
    contrast-enhanced only (not resampled), and the anatomical white-matter
    mask is resampled into that EPI space and drawn as a red contour. A red
    line that hugs the EPI's own gray/white boundary means accurate
    coregistration; a mask that is shifted/rotated relative to the EPI's own
    tissue boundaries means coregistration is failing.

    Returns a list of (label, fig) tuples.
    """
    figs = []
    for ses in sessions:
        task = run = None
        for t in tasks:
            runs = list_runs(sub, ses, t)
            if runs:
                task, run = t, runs[0]
                break
        if task is None:
            continue

        stub = func_stub(sub, ses, task, run)
        ref_path = func_dir(sub, ses) / f"{stub}_desc-coreg_boldref.nii.gz"
        xfm_matches = sorted(func_dir(sub, ses).glob(f"{stub}_from-boldref_to-T1w*_xfm.txt"))
        if not ref_path.exists() or not xfm_matches:
            continue
        xfm_path = xfm_matches[0]

        boldref_img = nib.load(str(ref_path))
        boldref = boldref_img.get_fdata(dtype=np.float32)
        if boldref.ndim == 4:
            boldref = boldref[..., 0]

        # T1w WM probseg resampled into this run's native EPI space
        # (fixed/reference = boldref, moving = WM mask defined in T1w space)
        wm_in_bold = resample_to_reference(
            moving_path=wm_path, reference_path=ref_path, xfm_path=xfm_path,
            invert=True, interpolator=sitk.sitkLinear)

        cx, cy, cz = center_slices(boldref)
        views = [
            (lambda v, i=cx: v[i, :, :].T, f"Sag x={cx}"),
            (lambda v, i=cy: v[:, i, :].T, f"Cor y={cy}"),
            (lambda v, i=cz: v[:, :, i].T, f"Axl z={cz}"),
        ]

        label = ses if ses else "sub"
        fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE, squeeze=False)
        fig.patch.set_facecolor("#111111")
        fig.suptitle(
            f"{sub}  |  BOLD → T1w coregistration  —  {label}  task-{task}"
            + (f"  run-{run}" if run else "") + "\n"
            "background = reference EPI (native space)  ·  "
            "red contour = T1w white-matter mask resampled into EPI space",
            color="white", fontsize=12)

        for col, (slc_fn, view_title) in enumerate(views):
            bold_slc = slc_fn(boldref)
            wm_slc   = slc_fn(wm_in_bold)

            axes[0, col].imshow(contrast_enhance(bold_slc), origin="lower",
                                cmap="gray", interpolation="nearest")
            axes[0, col].contour(wm_slc, levels=[0.5], colors="red", linewidths=0.8)
            axes[0, col].set_title(view_title, color="white", fontsize=10)
            axes[0, col].axis("off")

        plt.tight_layout()
        figs.append((label, fig))

    return figs


def make_surfaces_fig(sub, anat_dir):
    """
    Cortical surface quality: ribbon mask + pial/white contours on a dense
    coronal mosaic.

    The ribbon mask (desc-ribbon_mask) is the voxel-space gray-matter ribbon
    between the white and pial surfaces — the direct product of the surface
    reconstruction.  Overlaying it in orange on the T1w lets you instantly
    spot a missed gyrus: wherever the T1w shows a cortical fold with no orange
    ribbon, the surface failed to capture that gyrus.  Pial (red) and white
    (cyan) contour dots add a second verification layer.
    """
    t1w_path    = find_anat_file(anat_dir, sub, "desc-preproc_T1w.nii.gz")
    ribbon_path = find_anat_file(anat_dir, sub, "desc-ribbon_mask.nii.gz")
    brain_path  = find_anat_file(anat_dir, sub, "desc-brain_mask.nii.gz")

    t1w_img    = nib.load(str(t1w_path))
    ribbon_img = nib.load(str(ribbon_path))
    brain_img  = nib.load(str(brain_path))

    t1w    = t1w_img.get_fdata(dtype=np.float32)
    ribbon = ribbon_img.get_fdata(dtype=np.float32) > 0.5   # binary
    brain  = brain_img.get_fdata(dtype=np.float32) > 0.5
    aff    = t1w_img.affine

    pial_pts, white_pts = [], []
    for hemi in ("L", "R"):
        for name, store in (("pial", pial_pts), ("white", white_pts)):
            matches = sorted(p for p in anat_dir.glob(f"{sub}*hemi-{hemi}_{name}.surf.gii")
                              if "_space-" not in p.name)
            if matches:
                store.append(nib.load(str(matches[0])).darrays[0].data)
    pial  = np.vstack(pial_pts)  if pial_pts  else np.zeros((0, 3))
    white = np.vstack(white_pts) if white_pts else np.zeros((0, 3))

    # Coronal mosaic: 2 rows x 4 slices spanning the FULL brain y-range (from
    # the brain mask), not a fixed window, so the frontal and occipital poles
    # (common sites for a missed gyrus) are always included.
    nz_brain = np.argwhere(brain)
    y_min, y_max = nz_brain[:, 1].min(), nz_brain[:, 1].max()
    n_cols, n_rows = 4, 2
    margin = int(0.05 * (y_max - y_min))
    y_idx = np.linspace(y_min + margin, y_max - margin, n_cols * n_rows).astype(int)
    y_idx = np.clip(y_idx, 0, t1w.shape[1] - 1)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=FIG_SIZE)
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"{sub}  |  Cortical surface quality: ribbon mask (orange) on T1w\n"
        "orange = cortical ribbon mask  ·  orange dots = pial  ·  cyan = white matter",
        color="white", fontsize=11)

    for idx, (yi, ax) in enumerate(zip(y_idx, axes.flat)):
        t1_slc  = t1w[:, yi, :].T
        rib_slc = ribbon[:, yi, :].T

        ax.imshow(norm01(t1_slc), origin="lower", cmap="gray", interpolation="nearest")

        # Ribbon overlay in orange (semi-transparent)
        ov = np.zeros((*rib_slc.shape, 4))
        ov[rib_slc] = [1.0, 0.55, 0.0, 0.45]
        ax.imshow(ov, origin="lower", interpolation="nearest")

        # Surface contours
        surface_contour_on_slice(pial,  aff, axis=1, slice_idx=yi, ax=ax, color="orange")
        surface_contour_on_slice(white, aff, axis=1, slice_idx=yi, ax=ax, color="cyan")

        ax.set_title(f"y={yi}", color="white", fontsize=7)
        ax.axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Animated BOLD frames
# ---------------------------------------------------------------------------

def make_bold_clip_frames(sub, sessions, tasks):
    """
    Animated BOLD clip cycling through every run of every task in every
    session. Samples every N-th volume so that ~BOLD_TARGET_VOLS frames are
    shown per run while covering the full time-series.  Colour limits are
    fixed from the first available run for visual consistency.
    """
    run_list = []  # (ses, task, run)
    for ses in sessions:
        for t in tasks:
            for run in list_runs(sub, ses, t):
                run_list.append((ses, t, run))

    if not run_list:
        return []

    # Determine z-slices and colour limits from the first available run
    ses0, task0, run0 = run_list[0]
    p0 = func_dir(sub, ses0) / (
        f"{func_stub(sub, ses0, task0, run0)}"
        "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    if not p0.exists():
        return []
    data0, _ = load_bold(p0)
    brain0 = data0.mean(3) > 0.1 * data0.mean()
    _, _, cz = center_slices(brain0.astype(float))
    z_slices = np.clip([cz - 8, cz, cz + 8], 0, data0.shape[2] - 1)
    bv = data0[brain0]
    vmin, vmax = np.percentile(bv, 2), np.percentile(bv, 98)

    frames = []
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZE)
    fig.patch.set_facecolor("black")
    for ax in axes:
        ax.set_facecolor("black")

    for ses, task, run in run_list:
        p = func_dir(sub, ses) / (
            f"{func_stub(sub, ses, task, run)}"
            "_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
        if not p.exists():
            continue
        data, _ = load_bold(p)
        n_vols = data.shape[3]
        step = max(1, n_vols // BOLD_TARGET_VOLS)
        vol_indices = range(0, n_vols, step)
        label = ses if ses else "sub"
        for t in vol_indices:
            for zi, ax in zip(z_slices, axes):
                ax.clear()
                ax.imshow(data[:, :, zi, t].T, origin="lower", cmap="gray",
                          vmin=vmin, vmax=vmax, interpolation="nearest")
                ax.set_title(f"z={zi}", color="white", fontsize=9)
                ax.axis("off")
                ax.set_facecolor("black")
            fig.suptitle(
                f"{sub}  |  {label}  task-{task}" + (f"  run-{run}" if run else "") +
                f"   vol {t+1}/{n_vols}  (every {step}th)",
                color="white", fontsize=12)
            frames.append(fig_to_array(fig))

    plt.close(fig)
    return frames


# ---------------------------------------------------------------------------
# Per-subject frame sequence
# ---------------------------------------------------------------------------

def subject_frames(sub, tasks, max_sessions):
    print(f"\n{'='*60}\n  {sub}\n{'='*60}")
    frames = []

    print("  title …")
    fig = make_title_fig(sub)
    frames += hold(fig_to_array(fig), HOLD_TITLE)
    plt.close(fig)

    anat_dir = find_anat_dir(sub)
    sessions = list_functional_sessions(sub, tasks, max_sessions=max_sessions)
    print(f"  sessions: {sessions}")

    # (1) Cortical surface quality
    print("  surfaces …")
    try:
        fig = make_surfaces_fig(sub, anat_dir)
        frames += hold(fig_to_array(fig), HOLD_PANEL)
        plt.close(fig)
    except Exception as e:
        print(f"    skipped ({e})")

    # (2) Per-run motion & signal quality
    print("  run summary (all sessions) …")
    fig = make_run_summary_fig(sub, sessions, tasks)
    frames += hold(fig_to_array(fig), HOLD_PANEL)
    plt.close(fig)

    # (3) Temporal SNR — one slide per session
    print("  tSNR (all sessions) …")
    for label, fig in make_tsnr_fig(sub, sessions, tasks):
        print(f"    {label}")
        frames += hold(fig_to_array(fig), HOLD_PANEL)
        plt.close(fig)

    # (4) BOLD → T1w registration — one slide per session
    print("  registration (all sessions) …")
    try:
        wm_path = find_anat_file(anat_dir, sub, "label-WM_probseg.nii.gz")
        for label, fig in make_registration_fig(sub, sessions, tasks, wm_path):
            print(f"    {label}")
            frames += hold(fig_to_array(fig), HOLD_PANEL)
            plt.close(fig)
    except Exception as e:
        print(f"    skipped ({e})")

    # (5) Animated EPI volumes
    print("  BOLD clip (all sessions) …")
    frames += make_bold_clip_frames(sub, sessions, tasks)

    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS,
                   help=f"task labels to include (default: {DEFAULT_TASKS})")
    p.add_argument("--max_sessions", type=int, default=None,
                   help="optional cap on functional sessions per subject, "
                        "evenly sampled across all available sessions "
                        "(default: None, i.e. every session is included)")
    p.add_argument("--subs", nargs="*", default=None,
                   help="subjects to include (default: all found in FMRIPREP dir)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.subs:
        subjects = args.subs
    else:
        subjects = sorted(
            d.name for d in FMRIPREP.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        )
    print(f"Subjects: {subjects}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_tag   = "_".join(args.subs) if args.subs else "all"
    out_path  = OUT_DIR / f"{DATASET_NAME}_{sub_tag}_qc.mp4"

    # Collect all frames (list of HxWx3 uint8 arrays)
    all_frames = []
    for sub in subjects:
        all_frames.extend(subject_frames(sub, args.tasks, args.max_sessions))

    if not all_frames:
        print("No frames generated — check paths.")
        return

    print(f"\nWriting {len(all_frames)} frames → {out_path}")
    h, w = all_frames[0].shape[:2]

    fig_out, ax_out = plt.subplots(figsize=(w / DPI, h / DPI), dpi=DPI)
    fig_out.subplots_adjust(0, 0, 1, 1)
    ax_out.axis("off")

    im_obj = ax_out.imshow(all_frames[0])
    # CRF (constant-quality) instead of a fixed bitrate: bits go where the
    # content actually needs them (the animated BOLD clip) instead of being
    # spent at a flat rate through long static-panel holds. bitrate=-1 tells
    # ffmpeg to let -crf govern instead of forcing a -b:v target.
    writer = animation.FFMpegWriter(fps=FPS, bitrate=-1,
                                    extra_args=["-vcodec", "libx264", "-crf", "28",
                                                "-preset", "veryfast", "-pix_fmt", "yuv420p"])
    with writer.saving(fig_out, str(out_path), dpi=DPI):
        for frame in all_frames:
            im_obj.set_data(frame)
            writer.grab_frame()

    plt.close(fig_out)
    print(f"Done.  Output: {out_path}")


if __name__ == "__main__":
    main()

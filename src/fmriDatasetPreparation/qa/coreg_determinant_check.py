"""
Dataset-agnostic sanity check on fMRIPrep's per-run BOLD->T1w coregistration.

Background
----------
Every dataset's fmriprep run script in this repo passes --bold2t1w-dof 12
(full affine: independent scale + shear per axis), rather than fMRIPrep's own
documented default of 6 (rigid body: rotation + translation only). Because a
subject's head does not actually change size or shear between a functional
run and its own same-session anatomical, any scale != 1 in that transform is
a registration artifact, not anatomy. Under dof=12 the coregistration search
occasionally converges to a spurious affine fit for an individual run -
observed directly in BOLDMomentsDataset sub-05, where run-level coregistration
determinants ranged from a ~0.93 typical baseline up to 1.47 (47% spurious
volume inflation) on isolated runs scattered unpredictably across sessions.
This produces exactly the kind of brain-size-mismatch artifact visible in the
fmriprep QC movies, and does so silently - fMRIPrep does not fail or warn.

dof=12 does not miscalibrate every run, or even every subject (see MOSAIC
project notes) - this check exists to find which specific runs it did affect,
on any dataset, without having to eyeball QC movies frame by frame.

What this script does
----------------------
Recursively finds every fMRIPrep `*_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt`
under a given fmriprep derivatives directory (this filename is fMRIPrep's own
standard output naming, not something any per-dataset script customizes, so
this check works unmodified across all 9 MOSAIC source datasets and any
future one). For each run's transform it computes the 3x3 linear part's
determinant (net volume scaling) and per-axis singular values (anisotropy /
shear magnitude), then flags runs whose determinant deviates from that
*subject's own* median by more than --mad-thresh scaled MADs - a robust,
per-subject relative check, since the "normal" baseline scale can differ
across acquisition protocols without being wrong.

Usage
-----
  python coreg_determinant_check.py --fmriprep-dir $DATASETS_ROOT/BOLDMomentsDataset/derivatives/versionC/fmriprep
  python coreg_determinant_check.py --fmriprep-dir .../fmriprep --subs sub-05 --out-csv out.csv
"""
import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

XFM_GLOB = "*_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt"


def parse_affine_params(xfm_path):
    """Return the 3x3 linear part of an ITK affine .txt transform."""
    text = Path(xfm_path).read_text()
    m = re.search(r"Parameters:\s*([-0-9.eE ]+)", text)
    if m is None:
        return None
    vals = [float(x) for x in m.group(1).split()]
    if len(vals) < 9:
        return None
    return np.array(vals[:9]).reshape(3, 3)


def subject_from_path(path):
    m = re.search(r"(sub-[A-Za-z0-9]+)", path)
    return m.group(1) if m else "unknown"


def run_label_from_path(path):
    return os.path.basename(path).replace(
        "_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt", "")


def scan(fmriprep_dir, subs=None):
    xfm_paths = sorted(glob.glob(os.path.join(fmriprep_dir, "**", XFM_GLOB),
                                  recursive=True))
    rows = []
    for p in xfm_paths:
        sub = subject_from_path(p)
        if subs and sub not in subs:
            continue
        M = parse_affine_params(p)
        if M is None:
            continue
        det = float(np.linalg.det(M))
        singular_values = np.linalg.svd(M, compute_uv=False)
        rows.append(dict(
            subject=sub,
            run=run_label_from_path(p),
            determinant=det,
            sv_min=float(singular_values.min()),
            sv_max=float(singular_values.max()),
            anisotropy=float(singular_values.max() / singular_values.min()),
            path=p,
        ))
    return pd.DataFrame(rows)


def flag_outliers(df, mad_thresh=3.5):
    """Per-subject robust outlier flag on determinant (median + MAD)."""
    df = df.copy()
    df["flag"] = False
    df["robust_z"] = np.nan
    for sub, idx in df.groupby("subject").groups.items():
        vals = df.loc[idx, "determinant"]
        med = vals.median()
        mad = (vals - med).abs().median()
        # scale factor to make MAD consistent with std under normality
        robust_z = 0.6745 * (vals - med) / mad if mad > 0 else pd.Series(0, index=idx)
        df.loc[idx, "robust_z"] = robust_z
        df.loc[idx, "flag"] = robust_z.abs() > mad_thresh
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fmriprep-dir", required=True,
                   help="Path to a dataset's fmriprep derivatives directory "
                        "(e.g. $DATASETS_ROOT/<Dataset>/derivatives/<version>/fmriprep)")
    p.add_argument("--subs", nargs="*", default=None,
                   help="Restrict to these subject IDs (e.g. sub-05). Default: all found.")
    p.add_argument("--mad-thresh", type=float, default=3.5,
                   help="Robust z-score (median + MAD based) threshold for flagging a "
                        "run's coregistration determinant as an outlier relative to that "
                        "subject's other runs (default: 3.5)")
    p.add_argument("--out-csv", default=None,
                   help="Optional path to save the full per-run table as CSV")
    args = p.parse_args()

    df = scan(args.fmriprep_dir, subs=args.subs)
    if df.empty:
        print(f"No '{XFM_GLOB}' files found under {args.fmriprep_dir}")
        return

    df = flag_outliers(df, mad_thresh=args.mad_thresh)

    print(f"Scanned {len(df)} runs across {df['subject'].nunique()} subject(s) "
          f"under {args.fmriprep_dir}\n")
    for sub, sub_df in df.groupby("subject"):
        n_flag = sub_df["flag"].sum()
        med = sub_df["determinant"].median()
        print(f"{sub}: n_runs={len(sub_df)}  median_det={med:.3f}  "
              f"det_range=[{sub_df['determinant'].min():.3f}, {sub_df['determinant'].max():.3f}]  "
              f"flagged={n_flag}")
        if n_flag:
            for _, row in sub_df[sub_df["flag"]].iterrows():
                print(f"    ** {row['run']}: det={row['determinant']:.3f} "
                      f"(robust_z={row['robust_z']:.1f}, anisotropy={row['anisotropy']:.2f}x)")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nFull table written to {args.out_csv}")


if __name__ == "__main__":
    main()

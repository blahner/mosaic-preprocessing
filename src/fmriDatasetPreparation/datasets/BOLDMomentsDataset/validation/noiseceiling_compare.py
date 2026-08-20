"""
Single-subject, single-version noise-ceiling computation for BMD.

Numeric core of noiseceiling_bmd.ipynb, but scoped to one subject and one
derivatives version instead of looping over all 10 subjects and writing into
the shared $DATASETS_ROOT/MOSAIC/noiseceilings/ aggregate. Intended for
before/after comparisons when reprocessing a single subject (e.g. testing a
different fmriprep parameter) without touching the production noise-ceiling
files used by the rest of MOSAIC.

Output: derivatives/<version>/GLM/sub-XX/noiseceiling/
    sub-XX_BMD_phase-{task}_n-{n}_noiseceiling.npy
    sub-XX_BMD_phase-{task}_n-{n}_noiseceiling_flatmap.png
"""
from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hcp_utils as hcp
from nilearn import plotting

from src.utils.helpers import ComputeNoiseceiling


def plot_flatmap(noiseceiling, title, out_path, vmax=100):
    """Both-hemisphere flat map, matching noiseceiling_bmd.ipynb's style."""
    cortex_data_left = hcp.left_cortex_data(noiseceiling)
    cortex_data_right = hcp.right_cortex_data(noiseceiling)
    datamax = max(np.nanmax(cortex_data_left), np.nanmax(cortex_data_right))

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4),
                              subplot_kw={"projection": "3d"})
    plt.subplots_adjust(wspace=0)
    plotting.plot_surf(hcp.mesh.flat_left, cortex_data_left, threshold=1,
                        bg_map=hcp.mesh.sulc_left, colorbar=False, cmap="hot",
                        vmin=0, vmax=vmax, axes=axes[0])
    plotting.plot_surf(hcp.mesh.flat_right, cortex_data_right, threshold=1,
                        bg_map=hcp.mesh.sulc_right, colorbar=False, cmap="hot",
                        vmin=0, vmax=vmax, axes=axes[1])
    axes[0].invert_yaxis()
    axes[1].invert_yaxis()

    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="hot", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6)
    cbar.set_ticks([0, round(datamax), vmax])
    cbar.set_ticklabels([0, round(datamax), vmax])
    fig.suptitle(title)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(args):
    subject = f"sub-{int(args.subject):02}"
    dataset_root = os.path.join(os.getenv("DATASETS_ROOT"), "BOLDMomentsDataset")
    fmri_path = os.path.join(dataset_root, "derivatives", args.version, "GLM")
    save_root = os.path.join(fmri_path, subject, "noiseceiling")
    os.makedirs(save_root, exist_ok=True)

    n_task = {"train": [1, "avg"], "test": [1, "avg"]}
    results = {}

    for task in n_task:
        betas_path = os.path.join(
            fmri_path, subject, "prepared_betas",
            f"{subject}_organized_betas_task-{task}_normalized.pkl")
        with open(betas_path, "rb") as f:
            betas, stimorder = pickle.load(f)
        betas = betas.T  # -> (vertices, num_reps, num_stimuli)
        print(f"{subject} {args.version} task-{task} betas shape: {betas.shape}")

        for n in n_task[task]:
            ncsnr, noiseceiling = ComputeNoiseceiling(betas, n=n).compute_noiseceiling()
            noiseceiling[noiseceiling < 0] = 0
            out_path = os.path.join(
                save_root, f"{subject}_BMD_phase-{task}_n-{n}_noiseceiling.npy")
            np.save(out_path, noiseceiling)
            plot_flatmap(
                noiseceiling, f"{subject} {args.version} task-{task} n={n}",
                os.path.join(save_root,
                             f"{subject}_BMD_phase-{task}_n-{n}_noiseceiling_flatmap.png"))
            summary = dict(
                max=float(np.nanmax(noiseceiling)),
                median=float(np.nanmedian(noiseceiling)),
                mean=float(np.nanmean(noiseceiling)),
            )
            results[(task, n)] = summary
            print(f"  n={n}: max={summary['max']:.2f} median={summary['median']:.2f} "
                  f"mean={summary['mean']:.2f}  -> {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--subject", type=int, required=True,
                        help="Subject number 1-10")
    parser.add_argument("--version", default="versionC",
                        help="Derivatives version subdirectory to read prepared_betas "
                             "from / write noiseceiling to (default: versionC)")
    args = parser.parse_args()
    main(args)

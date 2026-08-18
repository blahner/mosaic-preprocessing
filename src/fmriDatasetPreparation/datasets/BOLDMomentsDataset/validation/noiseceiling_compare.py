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
"""
from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import pickle
import numpy as np

from src.utils.helpers import ComputeNoiseceiling


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

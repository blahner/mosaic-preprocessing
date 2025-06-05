import os
import numpy as np
from nilearn import plotting
import pickle
import matplotlib.pyplot as plt
import hcp_utils as hcp

def vectorized_correlation(x,y,dim=0):
    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, ddof=1, keepdims=True)
    y_std = y.std(axis=dim, ddof=1, keepdims=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr.ravel()

def list_rep(myList: list, reps: int):
    #returns a list of items in "mylist" that are repeated "reps" number of times
    repList = []
    # traverse for all elements
    for x in myList:
        if x not in repList: 
            count = myList.count(x)
            if count == reps:
                repList.append(x)
    return repList

class arguments():
    def __init__(self) -> None:
        self.root = "/data/vision/oliva/scratch/datasets/CC2017/video_fmri_dataset"
        self.task = "train" #"test" is not yet supported
        self.plot = True
        self.verbose = True
args = arguments()

nvertices = 91282
subject_maxvals = {'subject1': 0.86, 'subject2': 0.78, 'subject3': 0.73}

if args.task == 'train':
    numreps = 2

#generate unique pairs of repetition indices
pairs = []
for i in range(numreps):
    for j in range(i+1, numreps):
        pairs.append((i,j))

for sub_idx, sub in enumerate(range(1,4)):
    subject_intrasc = f"subject{sub}"
    fmri_data = np.zeros((len(pairs), nvertices))
    if args.verbose:
        print(f"loading betas from subject {sub}")
    #load raw ts estimates
    with open(os.path.join(args.root, "TSTrialEstimates", subject_intrasc, "estimates-prepared", "step01", f"{subject_intrasc}_z=0_TSTrialEstimates_task-{args.task}.pkl"), 'rb') as f:
        estimates_noz_test, condition_order = pickle.load(f) #shape nvideos, nvertices 

    repeated_conditions = list_rep(condition_order, numreps) #just get the conditions repeated exactly numpreps times
    estimates_noz_test_matrix = np.zeros((nvertices, numreps, len(repeated_conditions)))
    for count, cond in enumerate(repeated_conditions):
        idx = [i for i, c in enumerate(condition_order) if c == cond] #all indices of repeated condition "cond"
        estimates_noz_test_matrix[:,:,count] = estimates_noz_test[idx, :].T

    segments = {f"seg{s}": [] for s in range(1,19)}
    for idx, r in enumerate(repeated_conditions):
        seg = r.split('_')[0]
        segments[seg].append(idx)

    intraSC = 0
    for segment, idx in segments.items():
        intraSC += vectorized_correlation(estimates_noz_test_matrix[:, 0, idx].T, estimates_noz_test_matrix[:, 1, idx].T)

    intraSC = intraSC/len(segments) #average over segments

    if args.plot:
        views = ['lateral', 'medial'] #['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
        if args.verbose:
            print("plotting IntraSC results")
        plot_root = os.path.join(args.root, "IntraSC_TS", subject_intrasc)
        if not os.path.exists(plot_root):
            os.makedirs(plot_root)
        task = args.task
        intraSC[intraSC < 0] = 0 #treshold by 0
        stat = intraSC
        max_val = subject_maxvals[subject_intrasc] #np.nanmax(stat)
        #inflated brain
        for hemi in ['left','right']:
            mesh = hcp.mesh.inflated
            cortex_data = hcp.cortex_data(stat)
            bg = hcp.mesh.sulc
            for view in views:
                display = plotting.plot_surf_stat_map(mesh, cortex_data, hemi=hemi,
                threshold=0.01, bg_map=bg, view=view, vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.jpg"))
                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-inflated_hemi-{hemi}_view-{view}.svg"))
                plt.close()
            #flattened brain
            if hemi == 'left':
                cortex_data = hcp.left_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_left, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_left, colorbar=True, cmap='hot', vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-left.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-left.jpg"))
                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-left.svg"))
                plt.close()
            if hemi == 'right':
                cortex_data = hcp.right_cortex_data(stat)
                display = plotting.plot_surf(hcp.mesh.flat_right, cortex_data,
                threshold=0.01, bg_map=hcp.mesh.sulc_right, colorbar=True, cmap='hot', vmax=max_val)

                colorbar = display.axes[1]
                custom_ticks = np.array([0, np.nanmax(stat), max_val])
                custom_labels = [f"{tick:.2f}" for tick in custom_ticks]
                colorbar.set_yticks(custom_ticks)
                colorbar.set_yticklabels(custom_labels)

                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-right.png"), dpi=300)
                #plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-right.jpg"))
                plt.savefig(os.path.join(plot_root, f"{subject_intrasc}_intrasc_task-{task}_mesh-flat_hemi-right.svg"))
                plt.close()
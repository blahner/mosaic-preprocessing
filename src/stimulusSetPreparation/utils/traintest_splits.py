import os
import numpy as np

"""
compute train/test splits on the different fmri datasets. The test train splits will
be at the level of the run.
"""

def HCP_resting_traintest(project_root: str, dataset_root: str, doNotOverwrite: bool=True):
    """
    Compute train test run splits for the human connectome project resting state data.
    For each subject, randomly choose one of the four runs to be test. The other three are train.
    Input:
        project_root: path to root of project. store test/train splits here since they are not specific to the dataset itself
        dataset_root: path of HCP dataset root
        doNotOverwrite: bool. if true (default) this will prevent the test train splits from being overwritten
    Returns:
        None. Writes train text file and test text file listing the relative filepaths to the train and test runs
    """
    if doNotOverwrite:
        overwriteFlag=0
        if os.path.isfile(os.path.join(project_root, 'utils', 'traintest_splits', "hcp_resting_train.txt")):
            print(f"{os.path.join(project_root, 'utils', 'traintest_splits', 'hcp_resting_train.txt')} already exists")
            overwriteFlag+=1
        if os.path.isfile(os.path.join(project_root, 'utils', 'traintest_splits', 'hcp_resting_test.txt')):
            print(f"{os.path.join(project_root, 'utils', 'traintest_splits', 'hcp_resting_test.txt')} already exists")
            overwriteFlag+=1
        if overwriteFlag == 2: #skip the rest of the function, we already have the splits
            return None
        elif overwriteFlag == 1:
            raise ValueError("Only a train or test split was found for the HCP resting state data. Either both or none should have been found.")

    #load the subject list
    subjects = os.listdir(os.path.join(dataset_root, "pre-process", "hp2000_clean"))
    train = []
    test = []
    for sub in subjects:
        img_path = os.path.join(dataset_root, "pre-process", "hp2000_clean", str(sub), "MNINonLinear","Results")
        if not os.path.exists(img_path):
            continue
        run_folders = os.listdir(os.path.join(dataset_root, "pre-process", "hp2000_clean", str(sub), "MNINonLinear","Results"))
        #count how many valid run files are here
        valid_run_folders = []
        for folder in run_folders:
            tmp_file = os.path.join(str(sub), "MNINonLinear", "Results", folder, f"{folder}_Atlas_MSMAll_hp2000_clean.dtseries.nii")
            if os.path.isfile(os.path.join(dataset_root, "pre-process", "hp2000_clean", tmp_file)):
                valid_run_folders.append(folder)
        if len(valid_run_folders) < 1:
            continue
        #select one run from the available runs as the test run. The remaining are train
        possible_runs = list(range(len(valid_run_folders)))
        test_run = np.random.choice(possible_runs)
        train_run = np.delete(possible_runs, test_run)

        #append the file paths to the appropriate train/test list
        test_file = os.path.join(str(sub), "MNINonLinear", "Results", valid_run_folders[test_run], f"{valid_run_folders[test_run]}_Atlas_MSMAll_hp2000_clean.dtseries.nii")
        if os.path.isfile(os.path.join(dataset_root, "pre-process", "hp2000_clean", test_file)):
            test.append(test_file)
        else:
            raise ValueError(f"{test_file} is not a valid run file. Double check that no extraneous files or folders were added to this dataset path.")
        
        for tr in train_run:
            train_file = os.path.join(str(sub), "MNINonLinear", "Results", valid_run_folders[tr], f"{valid_run_folders[tr]}_Atlas_MSMAll_hp2000_clean.dtseries.nii")
            if os.path.isfile(os.path.join(dataset_root, "pre-process", "hp2000_clean", train_file)):
                train.append(train_file)
            else:
                raise ValueError(f"{train_file} is not a valid run file. Double check that no extraneous files or folders were added to this dataset path.")

        
    #if these assertions pass, then every file in the train and test splits exist, are not repeated, and the splits dont overlap
    train_test_intersection = list(set(train) & set(test))
    assert(len(set(train)) == len(train))
    assert(len(set(test)) == len(test))
    assert(len(train_test_intersection) == 0)

    #write train/test txt files
    with open(os.path.join(project_root, "utils", "traintest_splits", "hcp_resting_train.txt"), 'w') as f:
        for line in train:
            f.write(f"{line}\n")
    with open(os.path.join(project_root, "utils", "traintest_splits", "hcp_resting_test.txt"), 'w') as f:
        for line in test:
            f.write(f"{line}\n")
    
    return None

if __name__=="__main__":    
    np.random.seed(42)
    
    #prepare split for HCP resting state
    project_root = os.path.join("/data","vision","oliva","blahner","SheenBrain")
    hcp_dataset_root = os.path.join("/data","vision","oliva","scratch","datasets","hcp_dataset")

    HCP_resting_traintest(project_root, hcp_dataset_root, doNotOverwrite=True)



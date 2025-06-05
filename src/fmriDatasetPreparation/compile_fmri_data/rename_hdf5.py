import h5py

def rename_hdf5_phase_groups(filepath):
    with h5py.File(filepath, 'a') as f:
        # Get all groups that need renaming
        to_rename = []
        
        def visit_func(name):
            if '_phase-artificial_' in name:
                to_rename.append(name)
        
        # Collect all paths containing the pattern
        f.visit(visit_func)
        
        # Rename each found path
        for old_name in to_rename:
            new_name = old_name.replace('_phase-artificial_', '_phase-test_')
            if new_name not in f:
                print(f"Renaming {old_name} to {new_name}")
                f.move(old_name, new_name)
            else:
                print(f"Warning: Destination {new_name} already exists, skipping {old_name}")

# Usage
filepath = '/data/vision/oliva/datasets/MOSAIC/mosaic_version-1_0_0_renamed.hdf5'
rename_hdf5_phase_groups(filepath)
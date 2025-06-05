import h5py

def rename_hdf5_phase_groups_chunked(filepath):
    with h5py.File(filepath, 'a') as f:
        to_rename = []
        
        def visit_func(name):
            if '_phase-artificial_' in name:
                to_rename.append(name)
        
        # Collect all paths containing the pattern
        f.visit(visit_func)
        
        # Sort to ensure we handle parent groups before children
        to_rename.sort(key=lambda x: len(x.split('/')))
        
        # Rename each found path
        count = 0
        for old_name in to_rename:
            new_name = old_name.replace('_phase-artificial_', '_phase-test_')
            if new_name not in f:
                print(f"Renaming {old_name} to {new_name}")
                try:
                    f.move(old_name, new_name)
                    count += 1
                except Exception as e:
                    print(f"Error renaming {old_name}: {e}")
            else:
                print(f"Warning: Destination {new_name} already exists, skipping {old_name}")
        print(f"Renamed {count} files")

# Usage
filepath = '/data/vision/oliva/datasets/MOSAIC/mosaic_version-1_0_0_chunks_renamed.hdf5'
rename_hdf5_phase_groups_chunked(filepath)
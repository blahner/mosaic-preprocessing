from dotenv import load_dotenv
load_dotenv()
import h5py
import glob
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np

root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.

def aggregate_hdf5_files(input_dir, output_filepath):
    """
    Aggregate multiple individual HDF5 files into a single large one.
    Each subject becomes its own group in the aggregated file.
    
    Args:
        input_dir: Directory containing individual HDF5 files
        output_file: Path for the output aggregated HDF5 file
    """
    
    # Find all HDF5 files in the input directory
    hdf5_files = glob.glob(os.path.join(input_dir, "*.hdf5"))
    
    if not hdf5_files:
        raise ValueError(f"No HDF5 files found in {input_dir}")
    
    print(f"Found {len(hdf5_files)} HDF5 files to aggregate")
    
    # Create the aggregated file
    with h5py.File(output_filepath, 'w') as output_h5:
        nan_indices_all = set() #keep track of all nan indices across all subjects and datasets
        for hdf5_file in tqdm(hdf5_files, desc="Aggregating files"):
            # Extract subject ID from filename (format: sub-XX_DATASET_crc32-YYYYYYYY.hdf5)
            filename = Path(hdf5_file).stem
            full_filename = Path(hdf5_file).name
            
            # Extract subject_dataset part (everything before _crc32)
            if '_crc32-' in filename:
                subject_group_name = filename.split('_crc32-')[0]
            else:
                # Fallback if no crc32 in filename
                subject_group_name = filename
            
            # Open the individual file
            with h5py.File(hdf5_file, 'r') as input_h5:
                nan_indices_all.update(input_h5['nan_indices_all'])

                # Create a group for this subject in the output file
                subject_group = output_h5.create_group(subject_group_name)
                
                # Add the full filename (including crc32 hash) as an attribute
                subject_group.attrs['source_filename'] = full_filename
                
                # Copy all attributes from the root of the input file to the subject group
                for attr_name in input_h5.attrs.keys():
                    subject_group.attrs[attr_name] = input_h5.attrs[attr_name]
                
                # Copy all groups and datasets recursively
                def copy_group_recursive(src_group, dst_group):
                    """Recursively copy groups and datasets"""
                    for key in src_group.keys():
                        if isinstance(src_group[key], h5py.Group):
                            # Create new group and copy recursively
                            new_group = dst_group.create_group(key)
                            # Copy group attributes
                            for attr_name in src_group[key].attrs.keys():
                                new_group.attrs[attr_name] = src_group[key].attrs[attr_name]
                            copy_group_recursive(src_group[key], new_group)
                        
                        elif isinstance(src_group[key], h5py.Dataset):
                            # Copy dataset
                            dst_group.copy(src_group[key], key)
                
                # Copy all the data
                copy_group_recursive(input_h5, subject_group)
                
        output_h5.attrs.create('nan_indices_all', np.array(nan_indices_all)) #these nan indices are not ordered
    
    print(f"Successfully aggregated {len(hdf5_files)} files into {output_filepath}")

def main():
    # Example usage
    input_dir = "path/to/individual/hdf5/files"
    output_dir = "path/to/merged/hdf5/files"
    output_file = "mosaic_ram.hdf5"
    
    output_filepath = os.path.join(output_dir, output_file)

    aggregate_hdf5_files(input_dir, output_filepath)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate multiple HDF5 files into one")
    parser.add_argument("--input_dir", type=str, required=False, default=os.path.join(root_default, "hdf5_files", "single_subject"),
                       help="Directory containing individual HDF5 files")
    parser.add_argument("--output_dir", type=str, required=False, default=os.path.join(root_default, "hdf5_files", "merged"),
                       help="Directory containing individual HDF5 files")
    parser.add_argument("--output_file", type=str, required=False, default="mosaic_ram.hdf5",
                       help="Output filename for aggregated HDF5 file")
    
    args = parser.parse_args()
    
    output_filepath = os.path.join(args.output_dir, args.output_file)

    aggregate_hdf5_files(args.input_dir, output_filepath)
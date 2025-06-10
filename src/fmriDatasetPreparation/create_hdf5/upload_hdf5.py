from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import h5py
from pathlib import Path
import humanize
import boto3
from botocore.exceptions import NoCredentialsError
import h5py
import os
import zlib
import threading
from tqdm import tqdm

# Configuration
S3_BUCKET = os.environ.get('S3_BUCKET', 'your-s3-bucket-name')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
ALLOWED_EXTENSIONS = {'hdf5', 'h5'}

class TqdmUploadCallback:
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()
        
        # Create progress bar
        self.pbar = tqdm(
            total=self._size,
            unit='B',
            unit_scale=True,
            desc=f"Uploading {os.path.basename(filename)}"
        )

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            self.pbar.update(bytes_amount)

    def close(self):
        self.pbar.close()

# Modified upload function
def upload_with_progress(file_path, s3_client, bucket, key, metadata):
    """Upload file to S3 with progress tracking"""
    callback = TqdmUploadCallback(file_path)

    try:
        with open(file_path, 'rb') as f:
            s3_client.upload_fileobj(
                f,
                bucket,
                key,
                ExtraArgs={
                    'Metadata': metadata,
                    'ContentType': 'application/x-hdf'
                },
                Callback=callback
            )
        
        # Clean up progress bar if using tqdm
        if hasattr(callback, 'close'):
            callback.close()
            
        print(f"\n✅ Upload completed: {key}")
        
    except Exception as e:
        if hasattr(callback, 'close'):
            callback.close()
        raise e

def generate_content_hash(file_path, hash_length=8):
    """Generate a short hash based on file contents
    
    Args:
        file_path: Path to the file
        hash_length: Length of hash to return (4-8 characters recommended)
    
    Returns:
        Short hash string
    """
    # Option 1: CRC32 (fast, good distribution, 8 hex chars max)
    if hash_length <= 8:
        with open(file_path, 'rb') as f:
            crc = 0
            while True:
                chunk = f.read(8192)  # Read in chunks for memory efficiency
                if not chunk:
                    break
                crc = zlib.crc32(chunk, crc)
        # Convert to unsigned and take first hash_length characters
        return f"{crc & 0xffffffff:08x}"[:hash_length]

def extract_hdf5_metadata(file_path):
    """Extract metadata from HDF5 file"""
    metadata = {}
    try:
        with h5py.File(file_path, 'r') as f:
            # Try to get metadata from file attributes
            for key in ['dataset_name', 'subjectID', 'preprocessing_pipeline', 'owner_name', 
                       'owner_email', 'beta_pipeline', 'github_url', 'publication_url']:
                if key in f.attrs:
                    metadata[key] = f.attrs[key].decode('utf-8') if isinstance(f.attrs[key], bytes) else str(f.attrs[key])
                else:
                    metadata[key] = ""
    except Exception as e:
        print(f"Error reading HDF5 metadata: {e}")
        # Return empty metadata if file can't be read
        metadata = {
            'dataset_name': '',
            'subjectID': '',
            'preprocessing_pipeline': '',
            'owner_name': '',
            'owner_email': '',
            'beta_pipeline': '',
            'github_url': '',
            'publication_url': ''
        }
    
    return metadata

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def main(args):
    # Initialize S3 client
    try:
        s3_client = boto3.client('s3', region_name=AWS_REGION)
    except NoCredentialsError:
        print("AWS credentials not found. Please configure your credentials.")
        s3_client = None

    """Upload HDF5 file to S3 with metadata"""
    if not s3_client:
        raise ValueError("error: S3 client not configured)")
    
    if not allowed_file(Path(args.file_path).name):
        raise ValueError(f"error: Only HDF5 files {ALLOWED_EXTENSIONS} are allowed")
    
    try:
        metadata = extract_hdf5_metadata(file_path=args.file_path)
        
        size_human_readable = humanize.naturalsize(os.path.getsize(args.file_path))
        metadata.update({"file_size": size_human_readable})
        
        #use crc32 to hash hdf5 file. any change in file creates new hash
        #hash just intended to ensure files are same or different
        crc32_hash = generate_content_hash(args.file_path)

        filename = Path(args.file_path).name
        name, ext = os.path.splitext(filename)
        unique_filename = f"{name}_crc32-{crc32_hash}{ext}"

        metadata.update({"crc32_hash": crc32_hash})

        upload_with_progress(
            file_path=args.file_path,
            s3_client=s3_client,
            bucket=S3_BUCKET,
            key=unique_filename,
            metadata=metadata
        )

    except Exception as e:
        raise ValueError(f"{e}")

        
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="Path to hdf5 file to upload.")

    args = parser.parse_args()
    
    main(args)
from dotenv import load_dotenv
load_dotenv()
from PIL import Image
import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
import glob
import pandas as pd
import json
import argparse
import h5py
import shutil

"""
This script copies the stimulus image and video files from the individual datasets into the central
compiled dataset folder. All raw jpg and mp4 files are copied into the 'raw' folder. In the case of 
mp4 videos (for BMD and HAD), the folder named after the video's name that contains the video's frames
are copied to the 'frames' folder. The middle frame of each video is also copied to the 'middle_frames'
folder. If applicable, all filenames and folders in the compiled dataset directory are changed to use the 
stimuli's source filename. For example, BMD videos XXXX.mp4 are changed to use the original moments in time
filename. This usage corresponds to how the 'test.json' and 'train.json' files reference the stimuli.

Note that this script copies all stimuli from each dataset into the central folder without checking for duplicates (
duplicates should be overwritten). Thus, you should double check that the central dataset folder contains the appropriate
number of total stimuli. In the case of cropped stimuli (e.g., NSD slighly crops coco images from their original), pay attention 
to the order that the stimulus sets are compiled (see the associated bash script that does the copying).
This compiled dataset does not distinguish between crops of the same stimulus. 
"""

def main(args):
    save_root = os.path.join(args.dataset_root, "MOSAIC")
    raw_dir = os.path.join(save_root, "stimuli", "raw")
    middleframes_dir = os.path.join(save_root, "stimuli", "frames_middle")
    allframes_dir = os.path.join(save_root, "stimuli", "frames")

    #housekeeping
    if args.dataset == 'NaturalScenesDataset':
        nsd_root = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata")
        
        stimuli_path = os.path.join(nsd_root,"nsd_stimuli.hdf5")
        print("loading nsd stimuli...")
        with h5py.File(stimuli_path, 'r') as hdf:
            # Assume that images are stored under a key, e.g., 'images'
            filepaths = hdf['imgBrick'][:]  # Load all images, rougly 39 gb in size. call the variable filepaths for compatibility with other datasets
        assert(filepaths.shape[0] == 73000)
        
        #load info to get nsd filenames
        nsd_csv = pd.read_csv(os.path.join(nsd_root, "nsd_stim_info_merged.csv"))
        coco_annotation_val = COCO(annotation_file=os.path.join(nsd_root, "annotations_trainval2017", "annotations", "instances_val2017.json"))
        coco_annotation_train = COCO(annotation_file=os.path.join(nsd_root, "annotations_trainval2017", "annotations", "instances_train2017.json"))
        img_ids_val = coco_annotation_val.getImgIds()
        img_ids_train = coco_annotation_train.getImgIds()
        notshown = pd.read_csv(os.path.join(nsd_root, 'notshown.tsv'), header=None).values.flatten().tolist() #nsdIDs 1-indexed
        print ("Loading NSD stim block...")
        image_data_set = h5py.File(os.path.join(nsd_root, 'nsd_stimuli.hdf5'), 'r')
        print(image_data_set.keys())
        image_data = np.copy(image_data_set['imgBrick'])
        image_data_set.close()
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            if idx+1 in notshown: #notshown is 1-indexed. the idx+1 accounts for this. 
                continue #don't add the NSD images that were not shown. account for the 1 and 0 indexing difference
            cocoID = nsd_csv.loc[idx,'cocoId']
            #coco id to image filename
            if cocoID in img_ids_val:
                img_info = coco_annotation_val.loadImgs(int(cocoID))[0]
            elif cocoID in img_ids_train:
                img_info = coco_annotation_train.loadImgs(int(cocoID))[0]
            else:
                raise ValueError(f"cocoID {cocoID} not found in coco annotations files.")
            coco_filename = img_info['file_name']
            image = Image.fromarray(image_data[idx, :, :, :])
            image.save(os.path.join(raw_dir, coco_filename))
        
        #copy artificial stim
        filepaths = [os.path.join(nsd_root, "nsdsynthetic_jpg", f"{i:03}.jpg") for i in range(1,285)] #dont copy subject specific color images
        assert(len(filepaths) == 284)
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} artificial stimuli to central dataset"):
            destination_filename = Path(source_file).name
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)

    elif args.dataset == 'BOLD5000':
        stimuli_path_coco = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "COCO", "*.jpg")
        stimuli_path_imagenet = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "ImageNet", "*.JPEG")
        stimuli_path_scene_jpg = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "Scene", "*.jpg")
        stimuli_path_scene_jpeg = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "Scene_Stimuli", "Presented_Stimuli", "Scene", "*.jpeg")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_coco) + glob.glob(stimuli_path_imagenet) + glob.glob(stimuli_path_scene_jpg) + glob.glob(stimuli_path_scene_jpeg)
        assert(len(filepaths) == 4916)
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            filename = Path(source_file).name
            destination_filename = filename.replace('COCO_train2014_', '')
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)

    elif args.dataset == 'GenericObjectDecoding':
        stimuli_path_test = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "images", "test", "*.JPEG")
        stimuli_path_train = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "images", "training", "*.JPEG")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_test) + glob.glob(stimuli_path_train)
        assert(len(filepaths) == 1250)
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            destination_filename = Path(source_file).name
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)
    elif args.dataset == 'NaturalObjectDataset':
        stimuli_folders_imagenet = glob.glob(os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "imagenet","*"))
        #get all filepaths of the stimuli
        filepaths = []
        for folder in stimuli_folders_imagenet:
            filepaths.extend(glob.glob(os.path.join(folder, '*.JPEG')) + glob.glob(os.path.join(folder, '*.jpg')))
        
        stimuli_path_coco = os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "coco","*.jpg")
        filepaths.extend(glob.glob(stimuli_path_coco))
        assert(len(filepaths) == 57120)
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            destination_filename = Path(source_file).name
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)
    elif args.dataset == 'deeprecon':
        stimuli_path_artificial_image = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata","images", "ArtificialImage", "*.tiff")
        stimuli_path_letter_image = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata","images", "LetterImage", "*.tif")
        #the task stimuli is the exact same as GOD
        stimuli_path_test = os.path.join(args.dataset_root, "GenericObjectDecoding", "derivatives", "stimuli_metadata", "images", "test", "*.JPEG")
        stimuli_path_train = os.path.join(args.dataset_root, "GenericObjectDecoding", "derivatives", "stimuli_metadata", "images", "training", "*.JPEG")

        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path_test) + glob.glob(stimuli_path_train) + glob.glob(stimuli_path_artificial_image) + glob.glob(stimuli_path_letter_image)
        assert(len(filepaths) == 1300) #1200 train, 50 test, 40 artificial, 10 letter
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            destination_filename = Path(source_file).name
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)
    elif args.dataset == 'THINGS_fmri':
        stimuli_path = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "experimental_images","*.jpg")
        #get all filepaths of the stimuli
        filepaths = glob.glob(stimuli_path)
        assert(len(filepaths) == 8740)
        for idx, source_file in tqdm(enumerate(filepaths), total=len(filepaths), desc=f"copying {args.dataset} stimuli to central dataset"):
            destination_filename = Path(source_file).name
            destination_file = os.path.join(raw_dir, destination_filename)
            shutil.copy(source_file, destination_file)
    elif args.dataset == 'BOLDMomentsDataset':
        annotations = json.load(open(os.path.join(args.dataset_root, "BOLDMomentsDataset", "derivatives", "stimuli_metadata", "annotations.json"), 'r'))
        
        middleframe_path = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "frames_middle", "*.jpg")
        middleframe_filepaths = sorted(glob.glob(middleframe_path))
        assert(len(middleframe_filepaths) == 1102)
        for source_file in tqdm(middleframe_filepaths, total=len(middleframe_filepaths), desc=f"copying {args.dataset} middle frames to central dataset"):
            filename = Path(source_file).name
            filename_alias = filename.split('_')[0] #the XXXX video number
            mit_filename = filename.replace(filename_alias, Path(annotations[f"{filename_alias:04}"]['MiT_filename'].split('/')[-1]).stem)
            destination_file = os.path.join(middleframes_dir, mit_filename)
            shutil.copy(source_file, destination_file)
        
        allframe_folders = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "frames", "*")))
        assert(len(allframe_folders) == 1102)
        for source in tqdm(allframe_folders, total=len(allframe_folders), desc=f"copying video frames to central dataset for {args.dataset}."):
            foldername_alias = source.split('/')[-1]
            #change the source video filename to the Moments in Time filename
            mit_name = Path(annotations[f"{foldername_alias:04}"]['MiT_filename'].split('/')[-1]).stem
            destination = os.path.join(allframes_dir, mit_name)
            if not os.path.exists(destination):
                os.makedirs(destination)
            allframe_filepaths = sorted(glob.glob(os.path.join(source, "*.jpg")))
            for source_file in allframe_filepaths:
                filename = source_file.split('/')[-1]
                destination_file = os.path.join(destination, f"{mit_name}_{filename}")
                shutil.copy(source_file, destination_file)      

            for source_file in allframe_filepaths:
                filename = Path(source_file).name
                filename_alias = filename.split('_')[0] #the XXXX video number
                mit_filename = filename.replace(filename_alias, Path(annotations[f"{filename_alias:04}"]['MiT_filename'].split('/')[-1]).stem)
                destination_file = os.path.join(destination, mit_filename)
                shutil.copy(source_file, destination_file)

        mp4_folders = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset,"derivatives", "stimuli_metadata", "mp4_h264", "*")))
        assert(len(mp4_folders) == 1102)
        for source_file in tqdm(mp4_folders, total=len(mp4_folders), desc=f"copying mp4 files from {args.dataset}"):
            alias_name = Path(source_file.split('/')[-1]).stem
            destination_file = os.path.join(raw_dir, Path(annotations[f"{alias_name:04}"]['MiT_filename'].split('/')[-1]))
            shutil.copy(source_file, destination_file)
    elif args.dataset == 'HumanActionsDataset':
        middleframe_path = os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "frames_middle", "*.jpg")
        middleframe_filepaths = glob.glob(middleframe_path)
        assert(len(middleframe_filepaths) == 21600)
        for source_file in tqdm(middleframe_filepaths, total=len(middleframe_filepaths), desc=f"copying {args.dataset} middle frames to central dataset"):
            filename = source_file.split('/')[-1]
            destination_file = os.path.join(middleframes_dir, filename)
            shutil.copy(source_file, destination_file)
        allframe_folders = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset, "derivatives", "stimuli_metadata", "frames", "*")))
        assert(len(allframe_folders) == 180)
        for source in tqdm(allframe_folders, total=len(allframe_folders), desc=f"copying video frames to central dataset for {args.dataset}."):
            video_names = sorted(glob.glob(os.path.join(source, '*')))
            assert(len(video_names) == 120)
            for video in video_names:
                video_name = video.split('/')[-1] #lots of HAD videos have periods in the filename so Path(video).stem doesnt work
                destination = os.path.join(allframes_dir, video_name)
                if not os.path.exists(destination):
                    os.makedirs(destination)
                allframe_filepaths = sorted(glob.glob(os.path.join(video, '*.jpg')))
                for source_file in allframe_filepaths:
                    filename = source_file.split('/')[-1]
                    destination_file = os.path.join(destination, f"{video_name}_{filename}")
                    shutil.copy(source_file, destination_file)            

        mp4_folders = sorted(glob.glob(os.path.join(args.dataset_root, args.dataset, "Nifti", "stimuli", "*")))
        assert(len(mp4_folders) == 180)
        for video in tqdm(mp4_folders, total=len(mp4_folders), desc=f"copying mp4 files for {args.dataset}"):
            videos = glob.glob(os.path.join(video, '*.mp4'))
            assert(len(videos) == 120)
            for source_file in videos:
                filename = source_file.split('/')[-1]
                # Construct the full path of the source file
                destination_file = os.path.join(raw_dir,filename)
                shutil.copy(source_file, destination_file)

    else:
        raise ValueError(f"Dataset {args.dataset} not recognized.")

if __name__=='__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets")) #use default if DATASETS_ROOT env variable is not set.
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="The fmri dataset you want to copy.")
    parser.add_argument("--dataset_root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    args = parser.parse_args()
    
    main(args)
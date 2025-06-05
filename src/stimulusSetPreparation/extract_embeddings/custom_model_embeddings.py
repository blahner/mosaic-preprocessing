from dotenv import load_dotenv
load_dotenv()
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
import torch
import os
import torch.nn as nn
from tqdm import tqdm
import json
from pathlib import Path
import numpy as np
import argparse
import warnings
import yaml
from torch.utils.data import DataLoader
from torchvision.transforms import v2
warnings.simplefilter(action='ignore', category=FutureWarning)

#local
from src.utils.transforms import SelectROIs, ToTensorfMRI, ToTensorSubjectID, ToTensorAngleID, InverseNormalize
from src.utils.dataset import FMRIDataset
from src.utils.helpers import FilterDataset
from src.encoding_exp.encoding_utils.models.model import CNN, RegressionAlexNet

"""
Extract features from a randomly initialized or pretrained custom model.
This feature extraction script looks different from the other scripts (e.g., dreamsim)
because subject and datasets specific information (subjectID and visual angle) have to
be input to the model as well, so we can't simply grab all stimuli in the stimuli folder and 
pass it through.
"""

def main(args):
    # Load the config file
    with open(os.path.join(args.project_root, "Config", "brain_optimized", "config.yaml"), "r") as file:
        config = yaml.safe_load(file)
    device = torch.device("cuda" if config['device']['use_gpu'] and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model_tag = f"{config['project_name']}_{config['model']['type']}" #_best_model.pth"

    img_tsfm = v2.Compose([v2.Resize((config['stimulus']['input_resize'], config['stimulus']['input_resize'])),
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
    fmri_tsfm = v2.Compose([ToTensorfMRI(dtype='float32')])

    subjectID_tsfm = v2.Compose([ToTensorSubjectID()])
    angleID_tsfm = v2.Compose([ToTensorAngleID()])

    #load train and test jsons
    with open(os.path.join(args.root, 'train_naturalistic.json'), 'r') as f:
        train_val_naturalistic_all = json.load(f)
    with open(os.path.join(args.root, 'test_naturalistic.json'), 'r') as f:
        test_naturalistic_all = json.load(f)
    with open(os.path.join(args.root, 'test_artificial.json'), 'r') as f:
        test_artificial_all = json.load(f)

    dataset_preprocessing_train_val_naturalistic = FilterDataset(config['fmri']['subject_include'],
                                                    config['fmri']['dataset_include'],
                                                    config['fmri']['use_noiseceiling'])
    
    dataset_preprocessing_test_naturalistic = FilterDataset(config['fmri']['subject_include'],
                                                    config['fmri']['dataset_include'],
                                                    config['fmri']['use_noiseceiling'])

    dataset_preprocessing_test_artificial = FilterDataset(config['fmri']['subject_include'],
                                                    config['fmri']['dataset_include'],
                                                    config['fmri']['use_noiseceiling'])

    train_val_naturalistic, subjectID_mapping_train_val_naturalistic = dataset_preprocessing_train_val_naturalistic.filter_splits(train_val_naturalistic_all)
    test_naturalistic, subjectID_mapping_test_naturalistic = dataset_preprocessing_test_naturalistic.filter_splits(test_naturalistic_all)
    test_artificial, subjectID_mapping_test_artificial = dataset_preprocessing_test_artificial.filter_splits(test_artificial_all)

    subjectID_mapping_all = {**subjectID_mapping_train_val_naturalistic, **subjectID_mapping_test_naturalistic, **subjectID_mapping_test_artificial}
    all_subjects = list(subjectID_mapping_all.keys())
    assert len(set(subjectID_mapping_train_val_naturalistic.keys()) - set(subjectID_mapping_test_naturalistic.keys())) == 0, f"Trainng and testing subject filters should return the same set of subjects"

    #divid train set into train and val. test set is fixed separately
    train_val_ratio = config['data']['train_split'] #decimal for percent train
    shuffled_indices = np.random.permutation(len(train_val_naturalistic))
    cutoff = int(len(train_val_naturalistic)*train_val_ratio)
    train_indices = shuffled_indices[:cutoff].astype(int)
    val_indices = shuffled_indices[cutoff:].astype(int)

    train_naturalistic = [train_val_naturalistic[i] for i in train_indices]
    val_naturalistic = [train_val_naturalistic[i] for i in val_indices]

    ROI_selection = SelectROIs(selected_rois=config['fmri']['rois'])

    dataset_train_naturalistic = FMRIDataset(train_naturalistic, ROI_selection, config['fmri']['use_noiseceiling'], config['fmri']['trial_selection'], subjectID_mapping=subjectID_mapping_train_val_naturalistic, img_transforms=img_tsfm, fmri_transforms=fmri_tsfm, subjectID_transforms=subjectID_tsfm, angleID_transforms=angleID_tsfm)
    dataset_val_naturalistic = FMRIDataset(val_naturalistic, ROI_selection, config['fmri']['use_noiseceiling'], config['fmri']['trial_selection'], subjectID_mapping=subjectID_mapping_train_val_naturalistic, img_transforms=img_tsfm, fmri_transforms=fmri_tsfm, subjectID_transforms=subjectID_tsfm, angleID_transforms=angleID_tsfm)
    dataset_test_naturalistic = FMRIDataset(test_naturalistic, ROI_selection, config['fmri']['use_noiseceiling'], config['fmri']['trial_selection'], subjectID_mapping=subjectID_mapping_test_naturalistic, img_transforms=img_tsfm, fmri_transforms=fmri_tsfm, subjectID_transforms=subjectID_tsfm, angleID_transforms=angleID_tsfm)
    dataset_test_artificial = FMRIDataset(test_artificial, ROI_selection, config['fmri']['use_noiseceiling'], config['fmri']['trial_selection'], subjectID_mapping=subjectID_mapping_test_naturalistic, img_transforms=img_tsfm, fmri_transforms=fmri_tsfm, subjectID_transforms=subjectID_tsfm, angleID_transforms=angleID_tsfm)


    print("Number of Training samples:", len(dataset_train_naturalistic))
    print("Number of Validation samples:", len(dataset_val_naturalistic))
    print("Number of Testing samples:", len(dataset_test_naturalistic))
    print("Number of indices to predict:", len(ROI_selection.selected_roi_indices))

    dataloader_train_naturalistic = DataLoader(dataset_train_naturalistic, batch_size=1, shuffle=config['data']['shuffle'], num_workers=config['data']['num_workers'])
    dataloader_val_naturalistic = DataLoader(dataset_val_naturalistic, batch_size=1, shuffle=False, num_workers=config['data']['num_workers'])
    dataloader_test_naturalistic = DataLoader(dataset_test_naturalistic, batch_size=1, shuffle=False, num_workers=config['data']['num_workers'])
    dataloader_test_artificial = DataLoader(dataset_test_artificial, batch_size=1, shuffle=False, num_workers=config['data']['num_workers'])
    
    dataloaders = {'train_naturalistic': dataloader_train_naturalistic,'val_naturalistic': dataloader_val_naturalistic, 'test_naturalistic': dataloader_test_naturalistic, 'test_artificial': dataloader_test_artificial}

    num_vertices = len(ROI_selection.selected_roi_indices)
    print(f"number of vertices/classes: {num_vertices}")
    if config['model']['type'] == 'CNN':
        model = CNN(num_outputs=num_vertices, 
                    num_subjects=len(all_subjects), 
                    subject_embedding_dim=config['fmri']['subject_embedding_dim'],
                    angle_embedding_dim=config['fmri']['angle_embedding_dim']).to(device) #initialize model and put it on device
        print(f"There are {model.count_parameters()} trainable parameters, including biases")
    elif config['model']['type'] == "RegressionAlexNet":
        model = RegressionAlexNet(num_outputs=num_vertices, 
                    num_subjects=len(all_subjects), 
                    subject_embedding_dim=config['fmri']['subject_embedding_dim'],
                    angle_embedding_dim=config['fmri']['angle_embedding_dim'],
                    pretrained=config['model']['pretrained_AlexNet'],
                    bottleneck_dim=config['model']['bottleneck_dim']).to(device) #initialize model and put it on device
    else:
        raise ValueError(f"Model type {config['model']['type']} not recognized.")
    if config['training']['data_parallel']:
        model = nn.DataParallel(model)

    for weights in ['pretrained','random']:
        print(f"Using {weights} weights.")
        save_root = os.path.join(args.root, "model_features", "brain_optimized", f"{model_tag}_{weights}") #to save figures and other output
        if not os.path.exists(save_root):
            os.makedirs(save_root, exist_ok=True)

        #load pretrained weights
        if weights == 'pretrained':
            model.load_state_dict(torch.load(os.path.join(args.project_root, config['checkpoint']['save_dir'], f"{model_tag}_best_model.pth")))
        
        if isinstance(model, nn.DataParallel):
            train_nodes, eval_nodes = get_graph_node_names(model.module)
        else:
            train_nodes, eval_nodes = get_graph_node_names(model) #list of model layers or nodes

        model.eval()
        # get activations from all samples in the dataloader
        interested_layers = eval_nodes[17:] #['fc_combined', 'fc_output']#eval_nodes #["x","CNN.0","CNN.1","CNN.2","CNN.3","CNN.4","CNN.5","fc.0"] #layers I want the activations from. Must be in "nodes"
        for ilayer in interested_layers:
            assert ilayer in eval_nodes, f"Your speficied model layer {ilayer} was not found in the model. Layers for extraction must be one of {eval_nodes}."
        if isinstance(model, nn.DataParallel):
            feature_extractor = create_feature_extractor(model.module, return_nodes=interested_layers) #the term "feature" and "activation" is interchangeable here
        else:
            feature_extractor = create_feature_extractor(model, return_nodes=interested_layers) #the term "feature" and "activation" is interchangeable here

        for eval_set, dataloader in dataloaders.items():
            for d in tqdm(dataloader, total=len(dataloader)):
                filename = f"{Path(d['stimulus_filename'][0]).stem}_model-{model_tag}.npy"
                output_path = os.path.join(save_root, filename)
                if os.path.isfile(output_path):
                    continue #features already extracted for this stimulus
                features = {layer: np.array([]) for layer in interested_layers}
                stim = d['stimulus'].to(device)
                subjectID = d['subjectID'].to(device)
                viewing_angle = d['viewing_angle'].to(device)
                ft = feature_extractor(stim, subjectID, viewing_angle)
                for layer in interested_layers:
                    features[layer] = ft[layer][0].cpu().detach().numpy().flatten() #flatten
                np.save(output_path, features)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"), "MOSAIC") #use default if DATASETS_ROOT env variable is not set.
    project_root_default = os.path.join(os.getenv("PROJECT_ROOT", "/default/path/to/project"))

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='config.yaml', help="Configuration file that specifies model training.")
    parser.add_argument("-g", "--root", type=str, default=dataset_root_default, help="Root path to scratch datasets folder.")
    parser.add_argument("-i", "--pretrained", action="store_true", default=True, help="Whether to load pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained", help="Do not load pretrained weights")
    parser.add_argument("-p", "--project_root", type=str, default=project_root_default, help="Root path to project folder.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
        
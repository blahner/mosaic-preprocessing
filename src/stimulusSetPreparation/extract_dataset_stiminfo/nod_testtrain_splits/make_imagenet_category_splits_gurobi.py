from dotenv import load_dotenv
load_dotenv()
import os
import argparse
import random
import glob
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import gurobipy as gp
from gurobipy import GRB

"""
Inspired from this article: https://towardsdatascience.com/mathematics-of-love-optimizing-a-dining-room-seating-arrangement-for-weddings-with-python-f9c57cc5c2ce
Code: https://github.com/ceche1212/math_love_RQMKP_medium
You must set up an account with gurobi (https://www.gurobi.com/) to obtain licenses to run this. Academic account is free.
I put the license credentials in the projects .env file.
"""
def get_lowertriangular(rdm):
    num_conditions = rdm.shape[0]
    return rdm[np.triu_indices(num_conditions,1)]

def main(args):
    #set random seed
    random.seed(42)

    model_name = 'paraphrase-MiniLM-L6-v2'

    #load embeddings
    embedding_paths = sorted(glob.glob(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", f"imagenet_category_embeddings_{model_name}", "*.npy")))
    assert(len(embedding_paths) == 1000)
    #embedding_paths = embedding_paths[:50]
    embeddings = []
    category_label = []
    for embedding_path in embedding_paths:
        embeddings.append(np.load(embedding_path))
        filename = embedding_path.split('/')[-1]
        category_label.append(filename.split(f'_model-{model_name}_embedding.npy')[0])

    rdm = 1 - cosine_similarity(np.array(embeddings))
    ltrdm = get_lowertriangular(rdm) #get lower triangle

    lower_cutoff = np.percentile(ltrdm, 1/3*100) #values below this cutoff will get a compatibility score of -1, meaning they have a small distance to each other and are not diverse
    upper_cutoff = np.percentile(ltrdm, 2/3*100) #values above this cutoff will get a compatibility score of 1, meaning they have a large distance to each other and are diverse
    
    data = np.zeros(rdm.shape)
    data[rdm < lower_cutoff] = -1
    data[rdm > upper_cutoff] = 1

    # Define parameters (example guest list and compatibility values)
    col_to_idx = {label: i for i, label in enumerate(category_label)}
    #col_to_idx = {'Pikachu': 0, 'Meditite': 1, 'Machoke': 2, 'Tyrantrum': 3, 'Aipom': 4, 'Pelipper': 5}
    idx_to_col = {v: k for k, v in col_to_idx.items()}

    #Number of groupings and spots in each group
    #n_groups = 5 
    #B_k = [200, 200, 200, 200, 200]
    n_groups = 2
    B_k = [200,800]
    assert(n_groups == len(B_k))

    # Create a list of items and groups
    I = range(len(col_to_idx))  # items index set
    K = range(n_groups)  # groups index set
    x_vars = [(i,k) for i in I for k in K]

    #set E to specify pairs of items that need to be in the same group. 
    # Pairs are tuples of item index 
    # e.g., [(col_to_idx['Pikachu'],col_to_idx['Meditite']), 
    # (col_to_idx['Machoke'],col_to_idx['Tyrantrum'])].
    # Not applicable here so it is empty
    E = [] 

    #initialize gurobi solver
    params = {
    "WLSACCESSID": os.getenv('WLSACCESSID'),
    "WLSSECRET": os.getenv('WLSSECRET'),
    "LICENSEID": int(os.getenv('LICENSEID')),
    }
    env = gp.Env(params=params) #this line initializes the gurobi environment
    model = gp.Model('Max_Min_RQMKP',env=env) #initialize model

    X = model.addVars(x_vars,vtype=GRB.BINARY, name="X") # create variables
    Zeta = model.addVar(vtype=GRB.INTEGER, name="zeta") # create objective function variable

    # Objective function
    model.setObjective( Zeta, GRB.MAXIMIZE )

    # Constraint (1)
    model.addConstrs(gp.quicksum(X[i,k]*X[j,k]*data[(i,j)] for i in I for j in I) >= Zeta for k in K)

    # Constraint (2)
    model.addConstrs( gp.quicksum( X[i,k] for k in K ) == 1 for i in I)

    # Constraint (3)
    model.addConstrs( gp.quicksum( X[i,k] for i in I ) <= B_k[k]  for k in K)

    # Constraint (4)
    for i,j in E:
        for k in K:
            model.addConstr(X[i,k] == X[j,k])

    tl = 86400 #max runtime in seconds
    mip_gap = 0.05 #MIP gap in %, where 0 is optimal solution. 0.05 --> 5%

    model.setParam('TimeLimit', tl)
    model.setParam('MIPGap', mip_gap)
    model.optimize()

    #print out results and save
    max_min_groups = [ []  for k in K]
    for x in X.keys():
        if X[x].x > 0.95:
            guest_idx = x[0]
            group = x[1]
            guest_name = idx_to_col[guest_idx]
            max_min_groups[group].append(guest_idx)
            print(f"{guest_name} in table {group + 1}")

    groups_dict = {f"group_{k+1:02}": [] for k in range(len(max_min_groups))}
    Total_Z = 0
    for i,group in enumerate(max_min_groups):
        Z = 0
        group_comp = [idx_to_col[x] for x in group]
        for n1 in group:
            for n2 in group:
                Z += data[n1,n2]
        Total_Z += Z
        print(f" Group {i+1} Happiness (diversity) = {Z}")
        groups_dict[f"group_{i+1:02}"] = group_comp
        groups_dict.update({f"group_{i+1:02}_total_diversity": Z})
    print(f"Total Happiness (diversity) = {Total_Z}")

    #save groups
    with open(os.path.join(args.dataset_root, "derivatives", "stimuli_metadata", "imagenet_categories", "imagenet_category_groupings.pkl"), 'wb') as f:
        pickle.dump(groups_dict, f)

if __name__ == '__main__':
    dataset_root_default = os.path.join(os.getenv("DATASETS_ROOT", "/default/path/to/datasets"),"NaturalObjectDataset") #use default if DATASETS_ROOT env variable is not set.
    project_root_default = os.getenv("PROJECT_ROOT", "/default/path/to/datasets") #use default if DATASETS_ROOT env variable is not set.

    #arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root", default=dataset_root_default, help="The root path to the dataset directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-p", "--project_root", default=project_root_default, help="The root path to the project directory. Specifying path here overwrites environment variable.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose")

    args = parser.parse_args()

    main(args)
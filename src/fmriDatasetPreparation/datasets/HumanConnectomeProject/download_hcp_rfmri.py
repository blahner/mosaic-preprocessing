import os

sub_list = "./sub_list.txt"
output_dir = "./hp2000_clean/"

with open(sub_list, 'r') as f:
    sub_list = f.readlines()

sub_list = [sub.strip() for sub in sub_list if sub.strip().startswith('PRE')]
sub_list = [sub.split(' ')[-1][:-1] for sub in sub_list]

os.makedirs(output_dir, exist_ok=True)

for sub in sub_list:
    for area in ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']:
        os.makedirs(os.path.join(output_dir, sub, 'MNINonLinear/Results', area), exist_ok=True)
        os.system("aws s3 cp s3://hcp-openaccess/HCP_1200/%s/MNINonLinear/Results/%s/%s_Atlas_MSMAll_hp2000_clean.dtseries.nii %s" % (sub, area, area, os.path.join(output_dir, sub, 'MNINonLinear/Results', area)))


import os

# set paths
EXP_ROOT = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/audio_sheet_retrieval/"
DATA_ROOT_MSMD = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/msmd_aug_v1-1_no-audio/"
# EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
# DATA_ROOT_MSMD = '/media/matthias/Data/msmd_aug/'

# get hostname
hostname = os.uname()[1]

# adopted paths
if hostname == "Margaritas-MacBook-Pro.local":
    EXP_ROOT = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/audio_sheet_retrieval/"
    DATA_ROOT_MSMD = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/msmd_aug_v1-1_no-audio/"
        
elif hostname in ["rechenknecht0.cp.jku.at", "rechenknecht1.cp.jku.at"]:
    EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
    DATA_ROOT_MSMD = '/home/matthias/shared/datasets/msmd_aug/'

elif hostname == "mdhp":
    EXP_ROOT = "/home/matthias/experiments/audio_sheet_retrieval/"
    DATA_ROOT_MSMD = '/media/matthias/Data/Data/msmd/'

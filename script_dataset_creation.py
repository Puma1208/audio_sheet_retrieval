# from __future__ import print_function

from audio_sheet_retrieval.utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool
import sys
import yaml
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from msmd.data_model.piece import Piece


# config_file = '/home/I6194030/audio_sheet_retrieval/audio_sheet_retrieval/exp_configs/mutopia_no_aug.yaml'
# DATA_ROOT_MSMD = '/home/I6194030/msmd/msmd_aug_v1-1_no-audio'
# snippets_output_dir = "/home/I6194030/audio_sheet_retrieval/DATA_COLLECTION"

# split_file = '/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/msmd/msmd/splits/all_split.yaml'
config_file = '/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/audio_sheet_retrieval/audio_sheet_retrieval/exp_configs/mutopia_no_aug.yaml'
DATA_ROOT_MSMD = '/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/msmd_aug_v1-1_no-audio'
snippets_output_dir = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/audio_sheet_retrieval/DATA_COLLECTION_4"

DATA_ROOT_MSMD = '/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/testing_real_world_dataset/dataset'
snippets_output_dir = '/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/testing_real_world_dataset/piece_snippets'



# with open(config_file, 'rb') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)



SHEET_CONTEXT = 180
SYSTEM_HEIGHT = 160
SPEC_CONTEXT = 20
SPEC_BINS = 92


NO_AUGMENT = dict()
NO_AUGMENT['system_translation'] = 0
NO_AUGMENT['sheet_scaling'] = [1.00, 1.00]
NO_AUGMENT['onset_translation'] = 0
NO_AUGMENT['spec_padding'] = 0
NO_AUGMENT['interpolate'] = -1
NO_AUGMENT['synths'] = ['ElectricPiano']
NO_AUGMENT['tempo_range'] = [1.00, 1.00]

# this will be overwritten with a config file
# (see audio_sheet_retrieval/exp_configs)
AUGMENT = dict()
for key in NO_AUGMENT.keys():
    AUGMENT[key] = NO_AUGMENT[key]
    
# if not config_file:
spec_context = SPEC_CONTEXT
sheet_context = SHEET_CONTEXT
staff_height = SYSTEM_HEIGHT
augment = AUGMENT
no_augment = NO_AUGMENT
test_augment = NO_AUGMENT.copy()
# else:
#     with open(config_file, 'rb') as hdl:
#         config = yaml.load(hdl, Loader=yaml.FullLoader)
#     spec_context = config["SPEC_CONTEXT"]
#     sheet_context = config["SHEET_CONTEXT"]
#     staff_height = config["SYSTEM_HEIGHT"]
#     augment = config["AUGMENT"]
#     no_augment = NO_AUGMENT
#     test_augment = NO_AUGMENT.copy()
#     test_augment['synths'] = [config["TEST_SYNTH"]]
#     test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]


pieces_folder_names = os.listdir(DATA_ROOT_MSMD)
pieces_folder_names = ['Piece_4_Aube_rosee']
piece_pools = []

# Creatp
for i in tqdm(range(len(pieces_folder_names))):
# for i in tqdm(range(3, 4)):
    piece_name = pieces_folder_names[i]
    

    try:
        piece = Piece(root=DATA_ROOT_MSMD, name=piece_name)
        print(f'PIECE: {piece}')
        performances =  piece.available_performances
        if len(piece.available_scores)==0 or len(performances)==0:
            continue
        piece_image, piece_specs, piece_o2c_maps = prepare_piece_data(DATA_ROOT_MSMD, piece_name,
                                                    aug_config=augment, require_audio=False)
        piece_pool = AudioScoreRetrievalPool([piece_image], [piece_specs], [piece_o2c_maps], 
                                            spec_context=SPEC_CONTEXT, sheet_context=SHEET_CONTEXT, staff_height=SYSTEM_HEIGHT, 
                                            data_augmentation=test_augment, shuffle=False)
        piece_pools.append(piece_pool)


        snippets_piece_dir = os.path.join(snippets_output_dir, piece_name)
        sheet_snippets_dir = os.path.join(snippets_piece_dir, 'sheet_snippets')
        exceprt_snippets_dir = os.path.join(snippets_piece_dir, 'performances')

        # # iterate sheets 
        for i_sheet, sheet in enumerate(piece_pool.images):

            # iterate spectrograms
            for i_spec, spec in enumerate(piece_pool.specs[i_sheet]):
                sheet_dir = os.path.join(sheet_snippets_dir, performances[i_spec])
                exceprt_dir = os.path.join(exceprt_snippets_dir, performances[i_spec])
                
                for i_onset in range(len(piece_pool.o2c_maps[i_sheet][i_spec])):

                    snippet = piece_pool.prepare_train_image(i_sheet, i_spec, i_onset)
                    if not os.path.exists(sheet_dir):
                        os.makedirs(sheet_dir)
                    plt.imsave(f'{sheet_dir}/{performances[i_spec]}_{i_onset}.png', snippet)

                    excerpt = piece_pool.prepare_train_audio(i_sheet, i_spec, i_onset)
                    if not os.path.exists(exceprt_dir):
                        os.makedirs(exceprt_dir)
                    plt.imsave(f'{exceprt_dir}/{performances[i_spec]}_{i_onset}.png', excerpt)                

    except:
        print("Problems with loading piece %s" % piece_name)
        print(sys.exc_info()[0])
        continue







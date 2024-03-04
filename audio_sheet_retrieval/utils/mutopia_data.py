
from __future__ import print_function

import sys
import yaml
from tqdm import tqdm

from audio_sheet_retrieval.config.settings import DATA_ROOT_MSMD
from audio_sheet_retrieval.utils.data_pools import prepare_piece_data, AudioScoreRetrievalPool, AUGMENT, NO_AUGMENT
from audio_sheet_retrieval.utils.data_pools import SPEC_CONTEXT, SHEET_CONTEXT, SYSTEM_HEIGHT


def load_split(split_file):

    with open(split_file, 'rb') as hdl:
        split = yaml.load(hdl, Loader=yaml.FullLoader)

    return split


def load_piece_list(piece_names, aug_config=NO_AUGMENT):
    """
    Collect piece data
    """
    all_images = []
    all_specs = []
    all_o2c_maps = []
    for ip in tqdm(range(len(piece_names)), ncols=70):
        piece_name = piece_names[ip]

        try:
            image, specs, o2c_maps = prepare_piece_data(DATA_ROOT_MSMD, piece_name,
                                                        aug_config=aug_config, require_audio=False)
        except:
            print("Problems with loading piece %s" % piece_name)
            print(sys.exc_info()[0])
            continue

        # keep stuff
        all_images.append(image)
        all_specs.append(specs)
        all_o2c_maps.append(o2c_maps)

    return all_images, all_specs, all_o2c_maps


def load_audio_score_retrieval(split_file, config_file=None, test_only=False):
    """
    Load alignment data
    """

    if not config_file:
        spec_context = SPEC_CONTEXT
        sheet_context = SHEET_CONTEXT
        staff_height = SYSTEM_HEIGHT
        augment = AUGMENT
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
    else:
        with open(config_file, 'rb') as hdl:
            config = yaml.load(hdl, Loader=yaml.FullLoader)
        spec_context = config["SPEC_CONTEXT"]
        sheet_context = config["SHEET_CONTEXT"]
        staff_height = config["SYSTEM_HEIGHT"]
        augment = config["AUGMENT"]
        no_augment = NO_AUGMENT
        test_augment = NO_AUGMENT.copy()
        test_augment['synths'] = [config["TEST_SYNTH"]]
        test_augment['tempo_range'] = [config["TEST_TEMPO"], config["TEST_TEMPO"]]

    # selected pieces
    split = load_split(split_file)

    # initialize data pools
    if not test_only:
        tr_images, tr_specs, tr_o2c_maps = load_piece_list(split['train'], aug_config=augment)
        tr_pool = AudioScoreRetrievalPool(tr_images, tr_specs, tr_o2c_maps,
                                          spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=augment, shuffle=True)
        print("Train: %d" % tr_pool.shape[0])

        va_images, va_specs, va_o2c_maps = load_piece_list(split['valid'], aug_config=no_augment)
        va_pool = AudioScoreRetrievalPool(va_images, va_specs, va_o2c_maps,
                                          spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                          data_augmentation=no_augment, shuffle=False)
        va_pool.reset_batch_generator()
        print("Valid: %d" % va_pool.shape[0])

    else:
        tr_pool = va_pool = None

    te_images, te_specs, te_o2c_maps = load_piece_list(split['test'], aug_config=test_augment)
    te_pool = AudioScoreRetrievalPool(te_images, te_specs, te_o2c_maps,
                                      spec_context=spec_context, sheet_context=sheet_context, staff_height=staff_height,
                                      data_augmentation=no_augment, shuffle=False)
    print("Test: %d" % te_pool.shape[0])

    return dict(train=tr_pool, valid=va_pool, test=te_pool, train_tag="")


if __name__ == "__main__":
    """ main """
    print(SHEET_CONTEXT)
    # import matplotlib.pyplot as plt
    # from audio_sheet_retrieval.models.mutopia_ccal_cont_rsz import prepare

    # def train_batch_iterator(batch_size=1):
    #     """ Compile batch iterator """
    #     from audio_sheet_retrieval.utils.batch_iterators import MultiviewPoolIteratorUnsupervised
    #     batch_iterator = MultiviewPoolIteratorUnsupervised(batch_size=batch_size, prepare=None, k_samples=None)
    #     return batch_iterator

    # data = load_audio_score_retrieval(split_file="/home/matthias/cp/src/msmd/msmd/splits/all_split.yaml",
    #                                   config_file="/home/matthias/cp/src/audio_sheet_retrieval/audio_sheet_retrieval/exp_configs/mutopia_no_aug.yaml",
    #                                   test_only=True)

    # bi = train_batch_iterator(batch_size=5)

    # iterator = bi(data["test"])

    # # show some train samples
    # import time

    # for epoch in xrange(1000):
    #     start = time.time()
    #     for i, (sheet, spec) in enumerate(iterator):

    #         plt.figure()
    #         plt.clf()

    #         plt.subplot(1, 2, 1)
    #         plt.imshow(sheet[0, 0], cmap="gray")
    #         plt.ylabel(sheet[0, 0].shape[0])
    #         plt.xlabel(sheet[0, 0].shape[1])
    #         # plt.colorbar()

    #         plt.subplot(1, 2, 2)
    #         plt.imshow(spec[0, 0], cmap="gray_r", origin="lower")
    #         plt.ylabel(spec[0, 0].shape[0])
    #         plt.xlabel(spec[0, 0].shape[1])
    #         # plt.colorbar()

    #         plt.show()

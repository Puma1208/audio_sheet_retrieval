import tqdm 
import torch 
from torch import nn
import numpy as np

import os
import cv2
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from torch.utils.data import Dataset
from scipy.spatial import distance_matrix

msmd_images_dir = 'home/I6194030/audio_sheet_retrieval/DATA_COLLECTION'
msmd_images_dir = "/Users/margarita/Documents/DKE Master/Master Thesis - Music Score Localisation/audio_sheet_retrieval/DATA_COLLECTION_3"




# MSMD dataset

class MSMD_Dataset(Dataset):
    def __init__(self, root_dir, snippets_subfolder='sheet_snippets', performances_subfolder='performances'):
        self.dir = root_dir
        self.snippets_subfolder = snippets_subfolder
        self.performances_subfolder = performances_subfolder

        # self.pieces_folder = self.listdir_not_ds_store()
        self.pieces_folders =  sorted(os.listdir(self.dir))
        self.len = len(self.pieces_folders)

        self.transform = transforms.ToTensor()

    def __len__(self):
        if self.len is None:
            self.len = os.listdir(self.dir)
        return self.len
    
    def __getitem__(self, piece_index, performance_index=0):
        piece_path = os.path.join(self.dir, self.pieces_folders[piece_index])

        snippets_dir = os.path.join(piece_path, self.snippets_subfolder)
        snippets_path = os.path.join(snippets_dir, sorted(os.listdir(snippets_dir))[performance_index])

        performances_dir = os.path.join(piece_path, self.performances_subfolder)
        performances_path  = os.path.join(performances_dir, sorted(os.listdir(performances_dir))[performance_index])

        
        snippets = []
        # TODO sort on the last element after '_' to get the right order numerically
        for im in sorted(snippets_path):
            snippet = cv2.imread(os.path.join(snippets_path, im), cv2.IMREAD_GRAYSCALE)
            snippets.append(self.transform(snippet))

        excerpts = []
        for im in sorted(performances_path):
            excerpt = cv2.imred(os.path.join(performances_path, im), cv2.IMREAD_GRAYSCALE)
            excerpts.append(self.transform(excerpt))
        
        
        label = torch.tensor(([1]))

        # return torch.stack(snippets), torch.stack(excerpts), label
        return snippets, excerpts, label


    # TODO define method that gets a snippets from other piece where the label would be 0

def collate_fn(batch):
    snippets_batch = [torch.stack(sample[0]) for sample in batch]
    excerpts_batch = [torch.stack(sample[1]) for sample in batch]
    return snippets_batch, excerpts_batch

msmd_dataset = MSMD_Dataset(root_dir=msmd_images_dir)
msmd_loader =  DataLoader(msmd_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for i, data in enumerate(msmd_loader):
    print(len(data))

# Network
class SheetPassageCNN(nn.Module):
  def __init__(self,
               input_channels:int,
               output_shape:int
              #  width:int=92,
              #  height:int=20
               ):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=24,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(24)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=24,
                  out_channels=48,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(48)
    )
    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=48,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_5 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=32,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(32)
    )
    self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 6 * 5,
                      out_features=output_shape)
    )



  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f'layer 1: {x.shape}')
    x = self.conv_block_2(x)
    # print(f'layer 2: {x.shape}')
    x = self.conv_block_3(x)
    # print(f'layer 3: {x.shape}')
    x = self.conv_block_4(x)
    # print(f'layer 4: {x.shape}')
    x = self.conv_block_5(x)
    # print(f'layer 5: {x.shape}')
    return self.fully_connected(x)
class SpectrogramPassageCNN(nn.Module):
  
  '''
  Might use the same as SheetPassageCNN but depends if can take different output shape
  '''
  def __init__(self,
               input_channels:int,
            #    hidden_units:int,
               output_shape:int,
            #    width:int=92,
            #    height:int=20
               ):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(in_channels=input_channels,
                  out_channels=24,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(24)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=24,
                  out_channels=48,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(48)
    )

    self.conv_block_3 = nn.Sequential(
        nn.Conv2d(in_channels=48,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_4 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=96,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(96)
    )
    self.conv_block_5 = nn.Sequential(
        nn.Conv2d(in_channels=96,
                  out_channels=32,
                  kernel_size=1,
                  stride=1,
                  padding=0,
                  ),
        nn.ELU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.BatchNorm2d(32)
    )

    # self.fully_connected = nn.Linear(32, 32)
    self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32 * 2 * 1,
                      out_features=output_shape)
    )


  def forward(self, x):
    x = self.conv_block_1(x)
    # print(f'RNN layer 1: {x.shape}')
    x = self.conv_block_2(x)
    # print(f'RNN layer 2: {x.shape}')
    x = self.conv_block_3(x)
    # print(f'RNN layer 3: {x.shape}')
    x = self.conv_block_4(x)
    # print(f'RNN layer 4: {x.shape}')
    x = self.conv_block_5(x)
    # print(f'RNN layer 5: {x.shape}')
    return self.fully_connected(x)
class GRU_NN(nn.Module):


    '''https://blog.floydhub.com/gru-with-pytorch/'''
    def __init__(self, 
                 input_channels:int,
                 hidden_units:int, 
                 output_shape:int,
                 n_layers:int,
                 drop_prob=.2):
        
        super().__init__()
        self.hidden_units=hidden_units
        self.n_layers=n_layers
        self.output_shape=output_shape
        self.gru = nn.GRU(input_size=input_channels, 
                          hidden_size=hidden_units,
                          num_layers=n_layers,
                          bias=False,
                          batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_units, output_shape)
        self.relu = nn.ReLU()


    def forward(self, x, h=None):
        if h is None:
            h = self.init_hidden(x.shape[0])
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out[:, -1]))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_units).zero_()
        return hidden
class CNN_GRU(nn.Module):
    def __init__(self, 
                 input_channels_score:int, 
                 input_channels_audio:int, 
                 output_channels_cnn_score:int,
                 output_channels_cnn_audio:int,
                 hidden_units_score:int,
                 hidden_units_audio:int,
                 output_channels:int, 
                 ):
        super().__init__()
        self.score_passage_cnn = SheetPassageCNN(input_channels=input_channels_score,
                                                 output_shape=output_channels_cnn_score)
        self.spectrogram_passage_cnn = SpectrogramPassageCNN(input_channels=input_channels_audio,
                                                             output_shape=output_channels_cnn_audio)
        self.score_passage_gru = GRU_NN(input_channels=output_channels_cnn_score,
                                        hidden_units=hidden_units_score,
                                        output_shape=output_channels,
                                        n_layers=1)
        self.spectrogram_passage_gru = GRU_NN(input_channels=output_channels_cnn_audio,
                                              hidden_units=hidden_units_audio,
                                              output_shape=output_channels,
                                              n_layers=1)
        
    def forward(self, score_snippets, audio_snippets):
        batch_size_score, seq_len_score, h_score, w_score = score_snippets.size()
        batch_size_audio, seq_len_audio, h_audio, w_audio = audio_snippets.size()

        data_score = score_snippets.view(batch_size_score*seq_len_score, 1, h_score, w_score)
        data_audio = audio_snippets.view(batch_size_audio*seq_len_audio, 1, h_audio, w_audio)
        # print(f'Score shape BEFORE: {data_score.shape} | {data_audio.shape}')
        score_cnn = self.score_passage_cnn(data_score)
        spectrogram_cnn = self.spectrogram_passage_cnn(data_audio)
        # print(f'Score shape AFTER : {score_cnn.shape} | {spectrogram_cnn.shape}')
        features_score = score_cnn.view(batch_size_score, seq_len_score, -1)
        features_audio = spectrogram_cnn.view(batch_size_audio, seq_len_audio, -1)
        # print(f'Score shape AFTER CNN : {features_score.shape} | {features_audio.shape}')

        score_cnn_gru = self.score_passage_gru(features_score)
        spectrogram_cnn_gru = self.spectrogram_passage_gru(features_audio)
        # print(f'Score shape AFTER CRNN : {(score_cnn_gru.shape)} | {spectrogram_cnn_gru.shape}')


        return score_cnn_gru, spectrogram_cnn_gru

class Contrastive_Loss(nn.Module):

    # https://towardsdatascience.com/a-friendly-introduction-to-siamese-networks-85ab17522942
    # https://jamesmccaffrey.wordpress.com/2022/03/17/yet-another-siamese-neural-network-example-using-pytorch/
    def __init__(self, margin=1.):
        super().__init__()
        self.margin = margin

    def forward(self, x, y, same_flag):
        cosine_distance = nn.functional.cosine_similarity(x, y, dim=0)
        loss = torch.mean((1-same_flag)*torch.pow(cosine_distance, 2) +
                          (same_flag)*torch.pow(torch.clamp(self.margin-cosine_distance, min=.0), 2))
        return loss

siamese_network = CNN_GRU(1, 1, 32, 32, 128, 128, 4)
criterion = Contrastive_Loss()
optimizer = torch.optim.Adam(siamese_network.parameters(), lr=.05, weight_decay=.005)

losses = []
loss_contrastive = 0
epochs = 100
for epoch in tqdm(range(1, epochs+1)):
    siamese_network.train()
    for i, data in enumerate(msmd_loader):
        snippets, exceprts, label = data
        optimizer.zero_grad()
        output_snippets, output_excerpts = siamese_network()
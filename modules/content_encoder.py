import torch.nn as nn 

import pdb

class LocalContentEncoder(nn.Module):  #Similar to GANWriting (Kang, arxiv'2020) --g1
    def __init__(self, char_embed_dim):
        super(LocalContentEncoder,self).__init__()

        #Local
        self.g1Encoder = nn.Sequential(nn.Conv1d(char_embed_dim, char_embed_dim,1,1,0),
                            nn.BatchNorm1d(char_embed_dim),
                            nn.ReLU(True),
                            nn.Conv1d(char_embed_dim, char_embed_dim,1,1,0),
                            nn.BatchNorm1d(char_embed_dim),
                            nn.ReLU(True),
                            nn.Conv1d(char_embed_dim, char_embed_dim,1,1,0))
        # #Global
        # self.g2Encoder = nn.Sequential(nn.Linear(max_word_length*char_embed_dim, g2Dim),
        #                     nn.BatchNorm1d(g2Dim),
        #                     nn.ReLU(True),
        #                     nn.Linear(g2Dim, g2Dim),
        #                     nn.BatchNorm1d(g2Dim),
        #                     nn.ReLU(True),
        #                     nn.Linear(g2Dim, g2Dim))
    
    def forward(self, input):
        input = input.permute(0,2,1)
        
        
        return self.g1Encoder(input)


class GlobalContentEncoder(nn.Module):
    def __init__(self, num_class, max_word_length, char_embed_dim, dim):
        super(GlobalContentEncoder,self).__init__()
        self.max_word_length = max_word_length
        self.char_embed_dim = char_embed_dim
        self.charEmbed = nn.Embedding(num_class, char_embed_dim)

        #Global
        self.g2Encoder = nn.Sequential(nn.Linear(max_word_length*char_embed_dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(True),
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(True),
                            nn.Linear(dim, dim))
    
    def forward(self, input):
        input = self.charEmbed(input)
        input = input.view(-1,self.max_word_length*self.char_embed_dim)
        return self.g2Encoder(input)

        

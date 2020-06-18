"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.content_encoder import LocalContentEncoder
from modules.word_generator import WordGenerator,MLP

import pdb

class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
    

class AdaINGen(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGen, self).__init__()
        
        # dim = params['dim']
        # style_dim = params['style_dim']
        # n_downsample = ((2,2),(2,2),(2,1),(2,1))
        # n_res = params['n_res']
        # activ = params['activ']
        # pad_type = params['pad_type']
        # mlp_dim = params['mlp_dim']

        # self.enc_style = StyleEncoder(4, input_dim, dim, style_dim, norm='none', activ=activ, pad_type=pad_type)
        """ Style Encoder """
        # if opt.FeatureExtraction == 'VGG':
        self.enc_style = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        # elif opt.FeatureExtraction == 'RCNN':
        #     self.enc_style = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        # elif opt.FeatureExtraction == 'ResNet':
        #     self.enc_style = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        # else:
        #     raise Exception('No FeatureExtraction module specified')
        
        #char embedding
        self.batch_max_length = opt.batch_max_length
        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, 3, res_norm='adain', activ='relu', pad_type='reflect')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*opt.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2):
        # reconstruct an image
        
        style = self.enc_style(images)
        
        g1content_1 = self.enc_content_g1(self.charEmbed(labels_1))
        g1content_2 = self.enc_content_g1(self.charEmbed(labels_2))

        #Merge style and local content
        styleWidth = style.shape[3]
        styleHeight = style.shape[2]
        mulWFactor = int(styleWidth/self.batch_max_length)
        
        g1content_reshape_1 = g1content_1.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)
        g1content_reshape_2 = g1content_2.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)

        latent_1 = torch.cat((style,g1content_reshape_1),dim=1)
        latent_2 = torch.cat((style,g1content_reshape_2),dim=1)

        images_recon_1 = self.decode(latent_1, self.charEmbed(labels_1).reshape(-1,self.char_embed_size*self.batch_max_length))
        images_recon_2 = self.decode(latent_2, self.charEmbed(labels_2).reshape(-1,self.char_embed_size*self.batch_max_length))

        return images_recon_1, images_recon_2

    def decode(self, input, labels):
        # decode content and style codes to an image
        
        adain_params = self.mlp(labels)
        self.assign_adain_params(adain_params, self.dec)
        images = self.dec(input)
        return images

    def assign_adain_params(self, adain_params, model):
        # assign the adain_params to the AdaIN layers in model
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]

    def get_num_adain_params(self, model):
        # return the number of AdaIN parameters needed by the model
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params

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
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.content_encoder import LocalContentEncoder, GlobalContentEncoder
from modules.word_generator import WordGenerator,MLP, Conv2dBlock, ResBlocks, ResStyleBlocks, LinearBlock


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
                F=opt.num_fiducial, I_size=(opt.ocr_imgH, opt.ocr_imgW), I_r_size=(opt.ocr_imgH, opt.ocr_imgW), I_channel_num=opt.ocr_input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(ocr_imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (ocr_imgH/16-1) -> 1

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
        
        if self.opt.ocr_input_channel == 1 and self.opt.input_channel == 3:
            #rgb2gray conversion
            input = (input[:,0,:,:]*0.21+input[:,1,:,:]*0.72+input[:,1,:,:]*0.07).unsqueeze(1)
        
        if input.shape[2]!=self.opt.ocr_imgH or input.shape[3]!=self.opt.ocr_imgW:
            input = F.interpolate(input,(self.opt.ocr_imgH,self.opt.ocr_imgW),mode='bicubic', align_corners=False)

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

class ModelV1(nn.Module):

    def __init__(self, opt):
        super(ModelV1, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.ocr_imgH, opt.ocr_imgW), I_r_size=(opt.ocr_imgH, opt.ocr_imgW), I_channel_num=opt.ocr_input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.ocr_input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(ocr_imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (ocr_imgH/16-1) -> 1

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

    def forward(self, input, text, is_train=True, returnFeat='pred', inAct=None):
        
        # if inAct == 'tanh':
        #     input = torch.tanh(input)
        if self.opt.ocr_input_channel == 1 and self.opt.input_channel == 3:
            #rgb2gray conversion
            input = (input[:,0,:,:]*0.21+input[:,1,:,:]*0.72+input[:,1,:,:]*0.07).unsqueeze(1)
        
        if input.shape[2]!=self.opt.ocr_imgH or input.shape[3]!=self.opt.ocr_imgW:
            input = F.interpolate(input,(self.opt.ocr_imgH,self.opt.ocr_imgW),mode='bicubic', align_corners=False)

        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)
        
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)
        if returnFeat == 'vis':
            return visual_feature

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        if returnFeat == 'seq':
            return contextual_feature.contiguous()

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
        if 'CTC' in opt.Prediction:
            self.batch_max_length = opt.batch_max_length
        else:
            self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol
        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)

        if styleFlag:
            return style
        
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

        return images_recon_1, images_recon_2, style

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

class AdaINGenV1(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGenV1, self).__init__()
        
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
        if 'CTC' in opt.Prediction:
            self.batch_max_length = opt.batch_max_length
        else:
            self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol
        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)

        if styleFlag:
            return style
        
        g1content_1 = self.enc_content_g1(self.charEmbed(labels_1))
        g1content_2 = self.enc_content_g1(self.charEmbed(labels_2))

        #Merge style and local content
        styleWidth = style.shape[3]
        styleHeight = style.shape[2]
        mulWFactor = int(styleWidth/self.batch_max_length)
        
        g1content_reshape_1 = g1content_1.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)
        g1content_reshape_2 = g1content_2.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)

        latent_1 = torch.cat((style,g1content_reshape_1),dim=1)
        latent_2 = torch.cat((style.detach(),g1content_reshape_2),dim=1)

        images_recon_1 = self.decode(latent_1, self.charEmbed(labels_1).reshape(-1,self.char_embed_size*self.batch_max_length))
        images_recon_2 = self.decode(latent_2, self.charEmbed(labels_2).reshape(-1,self.char_embed_size*self.batch_max_length))

        return images_recon_1, images_recon_2, style

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


class AdaINGenV2(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGenV2, self).__init__()
        
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
        if 'CTC' in opt.Prediction:
            self.batch_max_length = opt.batch_max_length
        else:
            self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol
        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)

        if styleFlag:
            return style
        
        g1content_1 = self.enc_content_g1(self.charEmbed(labels_1))
        g1content_2 = self.enc_content_g1(self.charEmbed(labels_2))

        #Merge style and local content
        styleWidth = style.shape[3]
        styleHeight = style.shape[2]
        mulWFactor = int(styleWidth/self.batch_max_length)
        
        g1content_reshape_1 = g1content_1.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)
        g1content_reshape_2 = g1content_2.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,styleWidth,self.char_embed_size).permute(0,3,1,2)

        latent_1 = torch.cat((style,g1content_reshape_1),dim=1)
        latent_2 = torch.cat((style.detach(),g1content_reshape_2),dim=1)

        images_recon_1 = self.decode(latent_1, self.charEmbed(labels_1).reshape(-1,self.char_embed_size*self.batch_max_length))
        images_recon_2 = self.decode(latent_2, self.charEmbed(labels_2).reshape(-1,self.char_embed_size*self.batch_max_length))

        return images_recon_1, images_recon_2, style

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

class AdaINGenV3(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGenV2, self).__init__()
        
        """ Style Encoder """
        self.enc_style = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        
        #char embedding
        if 'CTC' in opt.Prediction:
            self.batch_max_length = opt.batch_max_length
        else:
            self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol
        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        
        adainParamSize = self.get_num_adain_params(self.dec)
        # if adainParamSize%4 != 0:
        #     print('Error in adain param size. Should be divisible by 4')
        # self.mlp = MLP(opt.char_embed_size*self.batch_max_length, int(adainParamSize/2), 1024, 3, norm='bn', activ='relu')
        self.mlp = MLP((opt.char_embed_size*self.batch_max_length)+opt.output_channel, adainParamSize, 1024, 3, norm='bn', activ='relu')

        #ToDo:
        #style input hardcoded for VGG input of size 32x100; Remove hardcoding
        # self.styleMLP = MLP(opt.output_channel*2*25, int(adainParamSize/2), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)
        
        styleEmbedInput = F.adaptive_avg_pool2d(style,(1,1)).squeeze()
        # style.reshape(-1,style.shape[1]*style.shape[2]*style.shape[3])

        if styleFlag:
            return styleEmbedInput
        
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

        images_recon_1 = self.decode(latent_1, torch.cat((self.charEmbed(labels_1).reshape(-1,self.char_embed_size*self.batch_max_length), styleEmbedInput),dim=1))
        images_recon_2 = self.decode(latent_2, torch.cat((self.charEmbed(labels_2).reshape(-1,self.char_embed_size*self.batch_max_length), styleEmbedInput),dim=1))

        return images_recon_1, images_recon_2, styleEmbedInput

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

class AdaINGenV4(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGenV4, self).__init__()
        
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
        # if 'CTC' in opt.Prediction:
        #     self.batch_max_length = opt.batch_max_length
        # else:
        #     self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol

        #V4: no addition of go and eos symbol
        self.batch_max_length = opt.batch_max_length
        self.Prediction = opt.Prediction

        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)

        if styleFlag:
            return style
        
        if not('CTC' in self.Prediction):
            labels_1 = labels_1[:,1:-1]
            labels_2 = labels_2[:,1:-1]

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

        return images_recon_1, images_recon_2, style

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

class AdaINGenV5(nn.Module):
    # AdaIN auto-encoder architecture
    def __init__(self, opt):
        super(AdaINGenV5, self).__init__()
        
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
        # if 'CTC' in opt.Prediction:
        #     self.batch_max_length = opt.batch_max_length
        # else:
        #     self.batch_max_length = opt.batch_max_length+2  #adding go and eos symbol

        #V4: no addition of go and eos symbol
        self.batch_max_length = opt.batch_max_length
        self.Prediction = opt.Prediction

        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,1),(2,1),(2,2),(2,2))
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters
        #g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.batch_max_length, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, images, labels_1, labels_2, styleFlag=False):
        # reconstruct an image
        style = self.enc_style(images)

        if styleFlag:
            return style
        
        if not('CTC' in self.Prediction):
            labels_1 = labels_1[:,1:-1]
            labels_2 = labels_2[:,1:-1]

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

        return images_recon_1, images_recon_2, style

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


class StyleTensorEncoder(nn.Module):
    def __init__(self, n_downsample=3, n_res=4, input_dim=1, dim=64, norm='bn', activ='lrelu', pad_type='zero'):
        super(StyleTensorEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]

        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(2 * dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(2 * dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2

        # residual blocks
        self.model += [ResStyleBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)



class StyleLatentEncoder(nn.Module):
    def __init__(self, n_downsample=3, n_res=4, input_dim=1, dim=64, norm='bn', activ='lrelu', pad_type='zero'):
        super(StyleLatentEncoder, self).__init__()
        self.model = []
        self.model += [VGG_FeatureExtractor(3, 512)]
        # self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        # self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]

        # # downsampling blocks
        # for i in range(n_downsample):
        #     self.model += [Conv2dBlock(dim, 2 * dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        #     self.model += [Conv2dBlock(2 * dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        #     self.model += [Conv2dBlock(2 * dim, 2 * dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        #     dim *= 2

        # # residual blocks
        # self.model += [ResStyleBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d((1,1))]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class AdaIN_Tensor_WordGenerator(nn.Module):
    def __init__(self, opt):
        super(AdaIN_Tensor_WordGenerator, self).__init__()
        
        self.batch_max_length = opt.batch_max_length
        # self.Prediction = opt.Prediction

        self.char_embed_size = opt.char_embed_size
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        if 'Attn' in opt.Prediction:
            self.text_len = self.batch_max_length+2
        else:
            self.text_len = self.batch_max_length

        # content encoder
        self.enc_content_g1 = LocalContentEncoder(opt.char_embed_size)
        n_downsample = ((2,2),(2,2),(2,2))
        
        # word generator
        self.dec = WordGenerator(n_downsample, 2, 512+opt.char_embed_size, opt.input_channel, res_norm='adain', activ='relu', pad_type='zero')

        # MLP to generate AdaIN parameters and g2 Content encoding
        self.mlp = MLP(opt.char_embed_size*self.text_len, self.get_num_adain_params(self.dec), 1024, 3, norm='bn', activ='relu')

    def forward(self, style, labels):
        
        g1content = self.enc_content_g1(self.charEmbed(labels))
        
        #Merge style and local content
        styleWidth = style.shape[3]
        styleHeight = style.shape[2]
        mulWFactor = int(styleWidth/self.text_len)
        
        g1content_reshape = g1content.repeat(1,mulWFactor,styleHeight).transpose(1,2).reshape(-1,styleHeight,mulWFactor*self.text_len,self.char_embed_size).permute(0,3,1,2)
        g1content_reshape = F.pad(g1content_reshape,(0,styleWidth-mulWFactor*self.text_len,0,0),'replicate')
        latent = torch.cat((style,g1content_reshape),dim=1)

        images_recon = self.decode(latent, self.charEmbed(labels).reshape(-1,self.char_embed_size*self.text_len))

        return images_recon

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

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

class L2Norm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.sum(input ** 2, dim=1, keepdim=True) + 1e-8)

class Mixer(nn.Module):
    def __init__(self, opt, nblk=3, dim=512):
        super(Mixer, self).__init__()
        self.dim=dim
        self.charEmbed = nn.Embedding(opt.num_class, opt.char_embed_size)

        if 'Attn' in opt.Prediction:
            self.text_len = opt.batch_max_length+2
        else:
            self.text_len = opt.batch_max_length
        self.NormLayer=PixelNorm()
        self.cEncoder = GlobalContentEncoder(opt.num_class, self.text_len, opt.char_embed_size, dim)
        # self.mixNet = MLP(dim*2, dim*2, dim*2, nblk, norm='none')
        self.mixNet = MLP(dim*2, dim, dim, nblk, norm='none')

    def forward(self, style, content):
        input = torch.cat([self.NormLayer(style),self.NormLayer(self.cEncoder(content))],dim=1)
        
        return self.mixNet(input)
        


class Fusion(nn.Module):
    def __init__(self, n_downsample=1, n_res=2, input_dim=512*2, dim=512, norm='bn', activ='lrelu', pad_type='zero'):
        super(Fusion, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]

        # downsampling blocks
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, dim, 3, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            self.model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activ, pad_type=pad_type)]

        # residual blocks
        self.model += [ResStyleBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

        self.linear_model = LinearBlock(4*16*dim, dim, norm='none', activation=activ)
        self.output_dim = dim

    def forward(self, styleCode, contentCode):
        
        x = self.model(torch.cat((styleCode,contentCode), dim=1))
        x = x.view(-1,4*16*self.output_dim)

        return self.linear_model(x)


class MsImageDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, opt):
        super(MsImageDis, self).__init__()
        self.n_layer = 3
        self.gan_type = 'lsgan'
        self.dim = 64
        self.norm = 'none'
        self.activ = 'lrelu'
        self.num_scales = 3
        self.pad_type = 'zero'
        self.input_dim = opt.input_channel
        self.downsample = nn.AvgPool2d(2, stride=1, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 3, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        real = input_real.data
        fake = input_fake.data
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0.data).cuda()
                all1 = torch.ones_like(out1.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class MsImageDisV1(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, opt):
        super(MsImageDisV1, self).__init__()
        self.n_layer = 3
        self.gan_type = opt.gan_type
        self.dim = 64
        self.norm = 'none'
        self.activ = 'lrelu'
        self.num_scales = 3
        self.pad_type = 'zero'
        self.input_dim = opt.input_channel
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 3, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_gradient_penalty(self, model, real_data, fake_data):
        LAMBDA = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = model(interpolates)

        # TODO: Make ConvBackward diffentiable
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        real = input_real.data
        fake = input_fake.data
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0.data).cuda()
                all1 = torch.ones_like(out1.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'wgan':
                
                model=self.cnns[it]
                loss += torch.mean(out0) - torch.mean(out1) + self.calc_gradient_penalty(model, real, fake)
                real = self.downsample(real)
                fake = self.downsample(fake)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'wgan':
                loss += -1 * torch.mean(out0)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

class MsImageDisV2(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, opt):
        super(MsImageDisV2, self).__init__()
        self.n_layer = 4
        self.gan_type = opt.gan_type
        self.dim = 64
        self.norm = 'none'
        self.activ = 'lrelu'
        self.num_scales = 3
        self.pad_type = 'zero'
        self.input_dim = opt.input_channel*2
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()
        for _ in range(self.num_scales):
            self.cnns.append(self._make_net())

    def _make_net(self):
        dim = self.dim
        cnn_x = []
        cnn_x += [Conv2dBlock(self.input_dim, dim, 3, 2, 1, norm='none', activation=self.activ, pad_type=self.pad_type)]
        for i in range(self.n_layer - 1):
            cnn_x += [Conv2dBlock(dim, dim * 2, 3, 2, 1, norm=self.norm, activation=self.activ, pad_type=self.pad_type)]
            dim *= 2
        cnn_x += [nn.Conv2d(dim, 1, 1, 1, 0)]
        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x

    def forward(self, x):
        outputs = []
        
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_gradient_penalty(self, model, real_data, fake_data):
        LAMBDA = 10
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        
        interpolates = interpolates.to(device)
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = model(interpolates)

        # TODO: Make ConvBackward diffentiable
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        real = input_real.data
        fake = input_fake.data
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0)**2) + torch.mean((out1 - 1)**2)
            elif self.gan_type == 'nsgan':
                all0 = torch.zeros_like(out0.data).cuda()
                all1 = torch.ones_like(out1.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            elif self.gan_type == 'wgan':
                
                model=self.cnns[it]
                loss += torch.mean(out0) - torch.mean(out1) + self.calc_gradient_penalty(model, real, fake)
                real = self.downsample(real)
                fake = self.downsample(fake)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1)**2) # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = torch.ones_like(out0.data).cuda()
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            elif self.gan_type == 'wgan':
                loss += -1 * torch.mean(out0)
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss


class VGGPerceptualLossModel(nn.Module):
    def __init__(self, vggmodel, criterion, resize=True):
        super(VGGPerceptualLossModel, self).__init__()
        
        self.layers = vggmodel.features
        
        self.layer_mapping = {
            '1':"relu1_1",
            '3':"relu2_1",
            '6':"relu3_1",
            '8':"relu4_1",
            '11':"relu5_1"
        }

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406])).view(1,3,1,1)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225])).view(1,3,1,1)

        self.resize = resize
        self.transform = torch.nn.functional.interpolate
        # self.normalize = torchvision.transforms.Normalize(mean,std)
        self.loss=criterion

    def forward(self, input, target, inAct=None, normFlag=True):
        pLoss = [] 
        sLoss = []

        # if inAct == 'tanh':
        #     input = torch.tanh(input)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        
        input = (input + 1) * 0.5
        target = (target + 1) * 0.5

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        
        #pre-process
        if normFlag:
            if input.is_cuda:
                input = (input-self.mean.cuda())/self.std.cuda()
                target = (target-self.mean.cuda())/self.std.cuda()
            else:
                input = (input-self.mean)/self.std
                target = (target-self.mean)/self.std
        

        for name,module in self.layers._modules.items():
            input = module(input)
            target = module(target)
            
            _,c,h,w = input.shape
            if name in self.layer_mapping:
                pLoss.append(self.loss(target, input))
                inputGram = (1/(h*w*c)) * torch.matmul(input.reshape(-1,c,h*w), input.reshape(-1,c,h*w).transpose(1,2))
                targetGram = (1/(h*w*c)) * torch.matmul(target.reshape(-1,c,h*w), target.reshape(-1,c,h*w).transpose(1,2))
                sLoss.append(self.loss(targetGram, inputGram))
              
        return torch.mean(torch.stack(pLoss)), torch.mean(torch.stack(sLoss))


class VGGPerceptualEmbedLossModel(nn.Module):
    def __init__(self, vggmodel, criterion, resize=True):
        super(VGGPerceptualEmbedLossModel, self).__init__()
        
        self.layers = vggmodel.features
        self.embed = vggmodel.classifier
        
        self.layer_mapping = {
            '1':"relu1_1",
            '3':"relu2_1",
            '6':"relu3_1",
            '8':"relu4_1",
            '11':"relu5_1"
        }

        self.embed_mapping = {
            '4':"relu2_1",
        }

        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406])).view(1,3,1,1)
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225])).view(1,3,1,1)

        self.resize = resize
        self.transform = torch.nn.functional.interpolate
        # self.normalize = torchvision.transforms.Normalize(mean,std)
        self.loss=criterion

    def forward(self, input, target, inAct=None, normFlag=True):
        pLoss = [] 
        sLoss = []
        eLoss = []

        # if inAct == 'tanh':
        #     input = torch.tanh(input)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        
        
        input = (input + 1) * 0.5
        target = (target + 1) * 0.5

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        
        
        #pre-process
        if normFlag:
            if input.is_cuda:
                input = (input-self.mean.cuda())/self.std.cuda()
                target = (target-self.mean.cuda())/self.std.cuda()
            else:
                input = (input-self.mean)/self.std
                target = (target-self.mean)/self.std
        
        
        for name,module in self.layers._modules.items():
            input = module(input)
            target = module(target)
            
            _,c,h,w = input.shape
            if name in self.layer_mapping:
                pLoss.append(self.loss(target, input))
                inputGram = (1/(h*w*c)) * torch.matmul(input.reshape(-1,c,h*w), input.reshape(-1,c,h*w).transpose(1,2))
                targetGram = (1/(h*w*c)) * torch.matmul(target.reshape(-1,c,h*w), target.reshape(-1,c,h*w).transpose(1,2))
                sLoss.append(self.loss(targetGram, inputGram))
        
        input = input.view((-1,2*8*512))
        target = target.view((-1,2*8*512))
        for name,module in self.embed._modules.items():
            input = module(input)
            target = module(target)
            
            if name in self.embed_mapping:
                eLoss.append(self.loss(target, input))
        
        return torch.mean(torch.stack(pLoss)), torch.mean(torch.stack(sLoss)), torch.mean(torch.stack(eLoss))


class VGGFontModel(nn.Module):
    def __init__(self, vggmodel, numClasses):
        super(VGGFontModel, self).__init__()
        

        self.features = vggmodel.features
        vggmodel.classifier[0]=torch.nn.Linear(2*8*512,4096)
        vggmodel.classifier[6]=torch.nn.Linear(4096,numClasses)
        self.classifier = vggmodel.classifier

        

    def forward(self, input):
        out = self.features(input)
        out = out.view((-1,2*8*512))
        out = self.classifier(out)
        return out
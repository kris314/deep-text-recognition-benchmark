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

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.content_encoder import LocalContentEncoder
from modules.word_generator import WordGenerator,MLP, Conv2dBlock


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
        # pdb.set_trace()
        for model in self.cnns:
            outputs.append(model(x))
            x = self.downsample(x)
        return outputs

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
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
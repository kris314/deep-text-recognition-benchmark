import os
import sys
import re
import six
import math
import lmdb
import random
import torch

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from utils import SynthGenerator

import phoc
import pdb

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(os.path.join(opt.exp_dir,opt.exp_name,'log_dataset.txt'), 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()
        
        self.pairText = opt.pairText
        self.lexicons=[]
        out_of_char = f'[^{opt.character}]'
        if opt.pairText == True:
            #read lexicons file
            with open(opt.lexFile,'r') as lexF:
                for line in lexF:
                    lexWord = line[:-1]
                    if opt.fixedString and len(lexWord)!=opt.batch_exact_length:
                        continue
                    if len(lexWord) <= opt.batch_max_length and not(re.search(out_of_char, lexWord.lower())) and len(lexWord) >= opt.batch_min_length:
                        self.lexicons.append(lexWord)
            

    def get_batch(self):
        
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                pdb.set_trace()
                image, text = data_loader_iter.next()
                
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        if self.pairText:
            return balanced_batch_images, balanced_batch_texts, random.sample(self.lexicons,len(balanced_batch_texts))
        else:
            return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                if opt.style_input:
                    if opt.style_content_input:
                        dataset = LmdbStyleContentDataset(dirpath, opt)
                    else:
                        dataset = LmdbStyleDataset(dirpath, opt)
                else:
                    dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log


class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print('nSamples:::::::::::::',nSamples)

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if self.opt.fixedString and len(label) != self.opt.batch_exact_length :
                        continue
                    elif len(label) > self.opt.batch_max_length or len(label)<self.opt.batch_min_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue
                    
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            
            label = txn.get(label_key).decode('utf-8')
            
            
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class LmdbStyleDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print('nSamples:::::::::::::',nSamples)

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label1_key = 'label1-%09d'.encode() % index
                    label1 = txn.get(label1_key).decode('utf-8')

                    label2_key = 'label2-%09d'.encode() % index
                    label2 = txn.get(label2_key).decode('utf-8')
                    
                    if self.opt.fixedString and (len(label1) != self.opt.batch_exact_length or len(label2) != self.opt.batch_exact_length) :
                        continue
                    elif len(label1) > self.opt.batch_max_length or len(label1)<self.opt.batch_min_length or len(label2) > self.opt.batch_max_length or len(label2)<self.opt.batch_min_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label1.lower()) or re.search(out_of_char, label2.lower()):
                        continue
                    
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label1_key = 'label1-%09d'.encode() % index
            label2_key = 'label2-%09d'.encode() % index
            
            label1 = txn.get(label1_key).decode('utf-8')
            label2 = txn.get(label2_key).decode('utf-8')
            
            
            img1_key = 'image1-%09d'.encode() % index
            img2_key = 'image2-%09d'.encode() % index
            img1buf = txn.get(img1_key)
            img2buf = txn.get(img2_key)

            buf1 = six.BytesIO()
            buf1.write(img1buf)
            buf1.seek(0)

            buf2 = six.BytesIO()
            buf2.write(img2buf)
            buf2.seek(0)

            try:
                if self.opt.rgb:
                    img1 = Image.open(buf1).convert('RGB')  # for color image
                    img2 = Image.open(buf2).convert('RGB')  # for color image
                else:
                    img1 = Image.open(buf1).convert('L')
                    img2 = Image.open(buf2).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img1 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img1 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label1 = '[dummy_label]'
                label2 = '[dummy_label]'

            if not self.opt.sensitive:
                label1 = label1.lower()
                label2 = label2.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label1 = re.sub(out_of_char, '', label1)
            label2 = re.sub(out_of_char, '', label2)

        return (img1, img2, label1, label2)

class LmdbStylePHOCDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        self.phocObj = phoc_gen(opt)

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print('nSamples:::::::::::::',nSamples)

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label1_key = 'label1-%09d'.encode() % index
                    label1 = txn.get(label1_key).decode('utf-8')

                    label2_key = 'label2-%09d'.encode() % index
                    label2 = txn.get(label2_key).decode('utf-8')
                    
                    if self.opt.fixedString and (len(label1) != self.opt.batch_exact_length or len(label2) != self.opt.batch_exact_length) :
                        continue
                    elif len(label1) > self.opt.batch_max_length or len(label1)<self.opt.batch_min_length or len(label2) > self.opt.batch_max_length or len(label2)<self.opt.batch_min_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label1.lower()) or re.search(out_of_char, label2.lower()):
                        continue
                    
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label1_key = 'label1-%09d'.encode() % index
            label2_key = 'label2-%09d'.encode() % index
            
            label1 = txn.get(label1_key).decode('utf-8')
            label2 = txn.get(label2_key).decode('utf-8')
            
            
            img1_key = 'image1-%09d'.encode() % index
            img2_key = 'image2-%09d'.encode() % index
            img1buf = txn.get(img1_key)
            img2buf = txn.get(img2_key)

            buf1 = six.BytesIO()
            buf1.write(img1buf)
            buf1.seek(0)

            buf2 = six.BytesIO()
            buf2.write(img2buf)
            buf2.seek(0)

            try:
                if self.opt.rgb:
                    img1 = Image.open(buf1).convert('RGB')  # for color image
                    img2 = Image.open(buf2).convert('RGB')  # for color image
                else:
                    img1 = Image.open(buf1).convert('L')
                    img2 = Image.open(buf2).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img1 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img1 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label1 = '[dummy_label]'
                label2 = '[dummy_label]'

            if not self.opt.sensitive:
                label1 = label1.lower()
                label2 = label2.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label1 = re.sub(out_of_char, '', label1)
            label2 = re.sub(out_of_char, '', label2)
            
        return (img1, img2, label1, label2, self.phocObj.getPhoc(label1), self.phocObj.getPhoc(label2))

class LmdbStyleContentDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        
        self.synthGen = SynthGenerator('/private/home/pkrishnan/fonts/fonts-10-path.txt',imgSize=(64,256))

        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            print('nSamples:::::::::::::',nSamples)

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label1_key = 'label1-%09d'.encode() % index
                    label1 = txn.get(label1_key).decode('utf-8')

                    label2_key = 'label2-%09d'.encode() % index
                    label2 = txn.get(label2_key).decode('utf-8')
                    
                    if self.opt.fixedString and (len(label1) != self.opt.batch_exact_length or len(label2) != self.opt.batch_exact_length) :
                        continue
                    elif len(label1) > self.opt.batch_max_length or len(label1)<self.opt.batch_min_length or len(label2) > self.opt.batch_max_length or len(label2)<self.opt.batch_min_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label1.lower()) or re.search(out_of_char, label2.lower()):
                        continue
                    
                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label1_key = 'label1-%09d'.encode() % index
            label2_key = 'label2-%09d'.encode() % index
            
            label1 = txn.get(label1_key).decode('utf-8')
            label2 = txn.get(label2_key).decode('utf-8')
            
            
            img1_key = 'image1-%09d'.encode() % index
            img2_key = 'image2-%09d'.encode() % index
            img1buf = txn.get(img1_key)
            img2buf = txn.get(img2_key)

            buf1 = six.BytesIO()
            buf1.write(img1buf)
            buf1.seek(0)

            buf2 = six.BytesIO()
            buf2.write(img2buf)
            buf2.seek(0)

            try:
                if self.opt.rgb:
                    img1 = Image.open(buf1).convert('RGB')  # for color image
                    img2 = Image.open(buf2).convert('RGB')  # for color image
                else:
                    img1 = Image.open(buf1).convert('L')
                    img2 = Image.open(buf2).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img1 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img1 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                    img2 = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label1 = '[dummy_label]'
                label2 = '[dummy_label]'

            if not self.opt.sensitive:
                label1 = label1.lower()
                label2 = label2.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label1 = re.sub(out_of_char, '', label1)
            label2 = re.sub(out_of_char, '', label2)
            label2SynthImg = self.synthGen.synthesizeWordImage(label2, 0)
            

        return (img1, img2, label1, label2, label2SynthImg)

class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        
        if self.max_size[2] != w:  # add border Pad
            # Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
            zero_vec = torch.zeros((img.size()[0],img.size()[1]))
            Pad_img[:, :, w:] = zero_vec.unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images, labels = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for image in images:
                w, h = image.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class AlignPairCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images1, images2, labels1, labels2 = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images1[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images1 = []
            resized_images2 = []
            cntr=0
            for image1 in images1:
                image2=images2[cntr]
                w, h = image1.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image1 = image1.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images1.append(transform(resized_image1))
                
                #resizing image2 w.r.t image1 dimensions
                resized_image2 = image2.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images2.append(transform(resized_image2))
                # resized_image.save('./image_test/%d_test.jpg' % w)
                cntr+=1

            image_tensors1 = torch.cat([t.unsqueeze(0) for t in resized_images1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in resized_images2], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors1 = [transform(image) for image in images1]
            image_tensors2 = [transform(image) for image in images2]
            image_tensors1 = torch.cat([t.unsqueeze(0) for t in image_tensors1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in image_tensors2], 0)

        return image_tensors1, image_tensors2, labels1, labels2

class AlignPHOCCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images1, images2, labels1, labels2, phoc1, phoc2 = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images1[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images1 = []
            resized_images2 = []
            cntr=0
            for image1 in images1:
                image2=images2[cntr]
                w, h = image1.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image1 = image1.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images1.append(transform(resized_image1))
                
                #resizing image2 w.r.t image1 dimensions
                resized_image2 = image2.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images2.append(transform(resized_image2))
                # resized_image.save('./image_test/%d_test.jpg' % w)
                cntr+=1

            image_tensors1 = torch.cat([t.unsqueeze(0) for t in resized_images1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in resized_images2], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors1 = [transform(image) for image in images1]
            image_tensors2 = [transform(image) for image in images2]
            image_tensors1 = torch.cat([t.unsqueeze(0) for t in image_tensors1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in image_tensors2], 0)
        
        phoc1_tensors = torch.cat([torch.tensor(t).unsqueeze(0) for t in phoc1], 0)
        phoc2_tensors = torch.cat([torch.tensor(t).unsqueeze(0) for t in phoc2], 0)

        return image_tensors1, image_tensors2, labels1, labels2, phoc1_tensors, phoc2_tensors

class AlignPairImgCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images1, images2, labels1, labels2, label2SynthImgs = zip(*batch)

        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = self.imgW
            input_channel = 3 if images1[0].mode == 'RGB' else 1
            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images1 = []
            resized_images2 = []
            resized_label2SynthImgs = []

            cntr=0
            for image1 in images1:
                image2=images2[cntr]
                label2SynthImg=label2SynthImgs[cntr]

                w, h = image1.size
                ratio = w / float(h)
                if math.ceil(self.imgH * ratio) > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = math.ceil(self.imgH * ratio)

                resized_image1 = image1.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images1.append(transform(resized_image1))
                
                #resizing image2 w.r.t image1 dimensions
                resized_image2 = image2.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images2.append(transform(resized_image2))

                resized_label2SynthImg = label2SynthImg.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_label2SynthImgs.append(transform(resized_label2SynthImg))
                # resized_image.save('./image_test/%d_test.jpg' % w)
                cntr+=1

            image_tensors1 = torch.cat([t.unsqueeze(0) for t in resized_images1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in resized_images2], 0)
            label2SynthImg_tensors = torch.cat([t.unsqueeze(0) for t in resized_label2SynthImgs], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors1 = [transform(image) for image in images1]
            image_tensors2 = [transform(image) for image in images2]
            
            label2SynthImg_tensors = [transform(image) for image in label2SynthImgs]
            image_tensors1 = torch.cat([t.unsqueeze(0) for t in image_tensors1], 0)
            image_tensors2 = torch.cat([t.unsqueeze(0) for t in image_tensors2], 0)
            label2SynthImg_tensors = torch.cat([t.unsqueeze(0) for t in label2SynthImg_tensors], 0)

        return image_tensors1, image_tensors2, labels1, labels2, label2SynthImg_tensors

def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# PHOC Dataset iterator
class phoc_gen(Dataset):
    def __init__(self, opt):
        
        unigram_levels = [1,2,3,4]
        bigram_levels = [2,3]

        self.word2Idx = {}
        self.idx2Word = {}
        self.words=[]
        strings=[]
        unqStrings=[]

        if os.path.exists(opt.words_file):
            lexFID = open(opt.words_file,'r')
            lines = lexFID.readlines()
            lexFID.close()
        else:
            sys.exit('Lexicon file not found')

        cntr=0
        gCntr=0
        for currWord in lines:
            if opt.sensitive:
                currWord = currWord[:-1]
            else:
                currWord = currWord[:-1].lower()
            
            if not(currWord in self.word2Idx):
                self.word2Idx[currWord] = cntr
                self.idx2Word[cntr] = currWord
                unqStrings.append(currWord)
                cntr+=1
            
            self.words.append(self.word2Idx[currWord])
            strings.append(currWord)
            gCntr+=1
        
        phoc_unigrams = opt.classes
        phoc_bigrams = phoc.get_most_common_n_grams(strings, num_results=50, n=2)
        
        self.phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
        
        if phoc_bigrams is not None:
            self.phoc_size += len(phoc_bigrams)*np.sum(bigram_levels)

        print('Building PHOC matrix--started')
        self.phocMat = np.zeros((len(self.words),self.phoc_size), dtype=np.float32)
        self.phocMat = phoc.build_phoc(unqStrings, phoc_unigrams, unigram_levels, 
            bigram_levels=bigram_levels, phoc_bigrams=phoc_bigrams, on_unknown_unigram='warn')
        print('Building PHOC matrix--ended')
    
    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        return (self.phocMat[self.words[index],:], self.idx2Word[self.words[index]])
    
    def getPhoc(self, word):
        if word in self.word2Idx:
            return self.phocMat[self.word2Idx[word],:]
        else:
            print('warning; phoc vector not found')
            return np.zeros((self.phoc_size),dtype=np.float32)
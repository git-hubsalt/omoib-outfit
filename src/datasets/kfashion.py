import os
import math
import datetime
import numpy as np

from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from dataclasses import dataclass

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from src.utils.utils import *
from src.datasets.processor import *

from PIL import Image


@dataclass
class DatasetArguments: 
    task_type: str = 'cp'
    dataset_type: str = 'train'

class KFashionDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        args: DatasetArguments,
        input_processor: FashionInputProcessor,
        ):

        # call arguments
        self.args = args
        self.data_dir = data_dir
        self.is_train = (args.dataset_type == 'train')

        self.img_dir = os.path.join(data_dir, 'image')
        self.input_processor = input_processor

        self.items, self.item_ids, self.categories = self.load_data(self.data_dir, self.args)
        self.data = self.fashion_cp_inputs(data_dir, args)

        self.length = len(self.data)


    def load_data(self, data_dir, args):
        meta_data_path = os.path.join(data_dir, 'item_metadata.json')
        meta_data = json.load(open(meta_data_path, encoding='utf-8'))

        item_ids, categories = set(), set()
        items = {}

        for item in meta_data:
            item_id = item['item_id']

            # check image path exists (전체 데이터 처리 후 삭제 예정) #
            img_path = os.path.join(self.img_dir, f"{item_id}.jpg")
            if not os.path.exists(img_path):
                continue
            # ----------------------------------------------------- #

            category = item['semantic_category']
            desc = item['title']
            style = item['style']

            items[item_id] = (category, desc, style) #
            item_ids.add(item_id)
            categories.add(category)

        return items, item_ids, categories


    def _load_img(self, item_id):
        path = os.path.join(self.img_dir, f"{item_id}")
        try:
            img = cv2.imread(path + ".jpg")
        except:
            img = cv2.imread(path + ".png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    

    def _get_inputs(self, item_ids: List[int], pad: bool=False) -> Dict[str, Tensor]:
        categories = [self.items[idx][0] for idx in item_ids]
        texts = [self.items[idx][1] for idx in item_ids]
        styles = [self.items[idx][2] for idx in item_ids]
        images = [self._load_img(idx) for idx in item_ids]

        return self.input_processor(categories, images, texts, styles, do_pad=pad)

    
    def __getitem__(self, idx):
        targets, outfits = self.data[idx]
        inputs = self._get_inputs(outfits, pad=True)
        
        return {'targets': targets, 'inputs': inputs}

    def __len__(self):
        return len(self.data)


    def fashion_cp_inputs(self, data_dir, args):
        cp_path = os.path.join(data_dir, f'compatibility_{args.dataset_type}.txt')
        cp_inputs = []

        with open(cp_path, 'r') as f:
            cp_data = f.readlines()
            for d in cp_data:
                item_ids = d.split()
                # check item exists (전체 데이터 처리 후 삭제) #
                if all(item_id in self.item_ids for item_id in item_ids):
                    cp_inputs.append((torch.FloatTensor([1]), item_ids))
                # ------------------------------------------------------- # 
                # cp_inputs.append((torch.FloatTensor([1]), item_ids))
        print(args.dataset_type, len(cp_inputs))
        return cp_inputs

    def load_sample(data_dir):
        item_ids, categories_ = set(), set()
        items = {}

        meta_data_path = os.path.join(data_dir, 'sample.txt')
        with open(meta_data_path, 'r') as file:
            samples = file.readlines()

            for item in samples:
                item_id = item['item_id']

                # check image path exists (전체 데이터 처리 후 삭제 예정) #
                img_path = os.path.join(self.img_dir, f"{item_id}.jpg")
                if not os.path.exists(img_path):
                    continue
                # ----------------------------------------------------- #

                category = item['semantic_category']
                desc = item['title']
                style = item['style']

                items[item_id] = (category, desc, style) #
                item_ids.add(item_id)
                categories.add(category)

        return items, item_ids, categories




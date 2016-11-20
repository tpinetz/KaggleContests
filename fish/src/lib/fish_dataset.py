from lib.base_dataset import Dataset
import os, glob
import lib.utils as utils
import numpy as np


class FishDataset(Dataset):
    def __init__(self, fdir, split):
        print('Read ' + split + ' images')
        
        data = []
        labels = []
        ids = []

        if split == 'test':
            path = os.path.join(fdir, '*.jpg')
            files = glob.glob(path)
            for fl in files:
                flbase = os.path.basename(fl)
                img = utils.get_im(fl)
                data.append(img.tolist())
                ids.append(flbase)
        else:
            folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
            for fld in folders:
                index = folders.index(fld)
                print('Load folder {} (Index: {})'.format(fld, index))
                path = os.path.join(fdir, fld, '*.jpg')
                files = glob.glob(path)
                
                split_idx = int(len(files)*0.8)
                if split == 'train':
                    files = files[:split_idx]
                elif split == 'val':
                    files = files[split_idx:]
                for fl in files:
                    flbase = os.path.basename(fl)
                    img = utils.get_im(fl)
                    data.append(img.tolist())
                    labels.append(index)
                    ids.append(flbase)
            
          
            
            Dataset.__init__(self, np.array(data, dtype=np.uint8), np.array(labels, dtype=np.uint8), ids)

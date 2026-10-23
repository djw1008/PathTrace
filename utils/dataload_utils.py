import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.append("..")


class geo_npz_Dataset_update(Dataset):
    """Load indexed methylation samples for training or evaluation."""
    
    def __init__(self, x_npy, index_file, if_train, 
                target_name="age",
                additional_list=["sample_id", "tissue"]):
        self.data = np.load(x_npy, allow_pickle=True)
        self.target_name = target_name
        self.additional_list = additional_list

        with open(index_file, 'rb') as f:   
            train_index = pickle.load(f)
            test_index = pickle.load(f)
        
        if if_train:
            self.key_list = train_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in train_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:
            self.key_list = test_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in test_index]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """Return the number of selected samples."""
        return len(self.key_list)

    def __getitem__(self, index):
        """Return one feature, age and metadata tuple."""
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"]
        age = self.data.item().get(key_name)["target"]
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_train(Dataset):
    """Load a named split from the PathTrace NPZ dataset."""
    
    def __init__(self, file_npy, data_type):
        data_npy = np.load(file_npy, allow_pickle=True)
        self.data = data_npy['data']

        if data_type == 'train':
            self.key_list = data_npy['train_index']
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in self.key_list]
            for test_name in move_list:
                del self.data.item()[test_name]
        elif data_type == 'val':
            self.key_list = data_npy['val_index']
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in self.key_list]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:
            self.key_list = data_npy['pretrain_index']
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in self.key_list]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """Return the number of selected samples."""
        return len(self.key_list)

    def __getitem__(self, index):
        """Return one feature, age and metadata tuple."""
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"].astype(np.float32)
        age = self.data.item().get(key_name)["target"].astype(np.float32)
        additional = self.data.item().get(key_name)["additional"]
        return feature, age, additional


class geo_npz_Dataset_inference(Dataset):
    """Load indexed methylation samples for inference."""
    
    def __init__(self, x_npy, index_file, if_train,
                 target_name="age",
                 additional_list=["sample_id", "tissue"]):
        self.data = np.load(x_npy, allow_pickle=True)
        self.target_name = target_name
        self.additional_list = additional_list

        with open(index_file, 'rb') as f:
            train_index = pickle.load(f)
            test_index = pickle.load(f)

        if if_train:
            self.key_list = train_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in train_index]
            for test_name in move_list:
                del self.data.item()[test_name]
        else:
            self.key_list = test_index
            all_key_list = self.data.item().keys()
            move_list = [value for value in all_key_list if value not in test_index]
            for test_name in move_list:
                del self.data.item()[test_name]

    def __len__(self):
        """Return the number of selected samples."""
        return len(self.key_list)

    def __getitem__(self, index):
        """Return one feature and metadata tuple."""
        key_name = self.key_list[index]
        feature = self.data.item().get(key_name)["feature"].astype(np.float32)
        additional = self.data.item().get(key_name)["additional"]
        return feature, additional


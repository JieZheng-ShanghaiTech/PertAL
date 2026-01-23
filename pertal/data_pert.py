import numpy as np
import torch
from .gears import PertData
import pickle
import os
from .gears.utils import dataverse_download

class Data:
    def __init__(self, path, data_name, 
                 batch_size, adata = None, test_fraction = 0.1, 
                 seed = 1, custom_test = None):
        
        
        pert_data = PertData(path) 
        if adata is not None:
            pert_data.new_data_process(dataset_name = data_name, adata = adata) 
            pert_data.load(data_path = os.path.join(path, data_name)) 
        else:
            if (not os.path.exists(path + data_name)):
                if data_name not in ['replogle_k562_essential_1000hvg','replogle_rpe1_essential_1000hvg',
                                     'adamson_k562_essential_1000hvg']:
                    raise ValueError('This dataset is not supported!')
                
                else:
                    server_path = 'https://dataverse.harvard.edu/api/access/datafile/7377294'
                    dataverse_download(server_path,
                           os.path.join(path, f'{data_name}.h5ad'))
                    import scanpy as sc
                    adata = sc.read_h5ad(os.path.join(path, f'{data_name}.h5ad'))
                    adata.uns['log1p']['base'] = None
                    pert_data.new_data_process(dataset_name = f'{data_name}', adata = adata)
            
            pert_data.load(data_path = path + data_name)
 
            split = 'active'
            test_perts = None

        pert_data.prepare_split(split = split,
                                test_perts = test_perts,
                                combo_single_split_test_set_fraction = test_fraction, 
                                seed = seed)

        self.batch_size = batch_size

        self.pert_data = pert_data
        self.pert_train = np.array(pert_data.set2conditions['train'])
        self.pert_test = np.array(pert_data.set2conditions['test'])
        
        self.n_pool = len(self.pert_train)
        self.n_test = len(self.pert_test)
        
        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)


    def initialize_labels(self, num):
        # Create an array of indices from 0 to the size of the pool (n_pool)
        tmp_idxs = np.arange(self.n_pool)
        np.random.seed(42)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
        return tmp_idxs[:num]
    
    def get_labeled_data(self, batch_exp = None):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        print('Number of labeled perts at this round: ' + str(len(labeled_idxs)))
       
        np.random.shuffle(labeled_idxs)        
        labeled_idxs_train = labeled_idxs[:int(0.9*len(labeled_idxs))]
        labeled_idxs_valid = labeled_idxs[int(0.9*len(labeled_idxs)):]

        return labeled_idxs, {'train_loader': self.pert_data.get_dataloader_from_pert_list(self.pert_train[labeled_idxs_train], 
                                                                                           self.batch_size, shuffle = True, 
                                                                                           eval_mode = False, batch_exp = batch_exp),
                              'val_loader': self.pert_data.get_dataloader_from_pert_list(self.pert_train[labeled_idxs_valid], 
                                                                                         self.batch_size, shuffle = False, 
                                                                                         eval_mode = True, batch_exp = batch_exp),
                              'valid_pert': self.pert_train[labeled_idxs_valid],
                              'train_pert': self.pert_train[labeled_idxs_train],
                              'train_valid_pert': self.pert_train[labeled_idxs]
                            }
    
    def get_unlabeled_data(self, get_distinct_perts = False):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        return unlabeled_idxs, self.pert_data.get_dataloader_from_pert_list(self.pert_train[unlabeled_idxs], self.batch_size, shuffle = False, eval_mode = True, get_distinct_perts = get_distinct_perts)
    
    def get_train_data(self, get_distinct_perts = False, batch_exp = None):
        return self.labeled_idxs.copy(), self.pert_data.get_dataloader_from_pert_list(self.pert_train, self.batch_size, 
                                                                                      shuffle = False, eval_mode = True, 
                                                                                      get_distinct_perts = get_distinct_perts,
                                                                                      batch_exp = batch_exp)

    def get_test_data(self, get_distinct_perts = False):
        return self.pert_data.get_dataloader_from_pert_list(self.pert_test, self.batch_size, shuffle = False, eval_mode = True, get_distinct_perts = get_distinct_perts)

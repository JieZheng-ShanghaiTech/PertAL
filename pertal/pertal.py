import numpy as np
import torch
import os
from .data_pert import Data
from .nets_pert import Net
import pickle
from .utils import load_kernel, get_strategy
from .gears.utils import zip_data_download_wrapper

class PertAL:

    def __init__(self,
                 weight_bias_track = False,
                 device = 'cuda',
                 proj_name = 'AL-PerturbSeq-Budget',
                 exp_name = 'PertAL',
                 seed = 1,
                 run = 1,
                 num_cpus = 4,
                 llm_weight=0.2,
                 gradient_weight=0.2
                 ):
        
        os.environ["OMP_NUM_THREADS"] = str(num_cpus) 
        os.environ["OPENBLAS_NUM_THREADS"] = str(num_cpus)
        os.environ["MKL_NUM_THREADS"] = str(num_cpus) 
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(num_cpus) 
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_cpus)
        
        
        if weight_bias_track:
            ## Initialize wandb with the project name, experiment name, and experiment configurations
            import swanlab as wandb
            wandb.init(project=proj_name, name=exp_name)  
            self.wandb = wandb
        else:
            self.wandb = None

        self.proj_name = proj_name
        self.exp_name = exp_name
        self.device = device
        self.seed = seed
        self.run = run
        self.llm_weight=llm_weight
        self.gradient_weight=gradient_weight

        np.random.seed(seed)
        torch.manual_seed(run)
        torch.backends.cudnn.enabled = False
        use_cuda = torch.cuda.is_available()
        self.device = torch.device(self.device if use_cuda else "cpu")
        print('device: ', device)

    def initialize_data(self, path, dataset_name, adata = None, batch_size = 256, 
                        test_fraction = 0.1, custom_test = None,llm_name=None):
        self.dataset = Data(path, dataset_name, batch_size, adata, test_fraction, self.seed, custom_test)
        self.test_data = self.dataset.get_test_data()
        self.dataset_name = dataset_name
        self.path = path
        self.llm_name=llm_name

    def initialize_model(self, epochs = 20, hidden_size = 64,
                                uncertainty = False,
                                uncertainty_reg = 1,
                                simple_loss = True,
                                direction_lambda = 1,
                                retrain = True,
                                fix_evaluation = True
                                ):
        params = {
            'weight_bias_track': self.wandb,
            'wb_proj_name': self.proj_name,
            'wb_exp_name': self.exp_name,
            'hidden_size': hidden_size,
            'uncertainty' : uncertainty, 
            'uncertainty_reg' : uncertainty_reg,
            'direction_lambda' : direction_lambda,
            'device': self.device,
            'epoch_per_cycle': epochs,
            'retrain': retrain,
            'simple_loss': simple_loss
        }
        
        self.net = Net(params, self.device, self.dataset.pert_data, fix_evaluation)                   # load network

    def initialize_active_learning_strategy(self, strategy,prior_scfm_kernel=None,agg_func=None):
        available = ['Random', 'BALD', 'BatchBALD', 'BAIT', 'ACS-FW', 'Core-Set', 'BADGE', 'LCMD', 'IterPert', 'TypiClust',
                      'KMeansSampling','PertAL']
        if strategy not in available:
            raise ValueError('Strategy not in the current available set: ' + ' '.join(available))

        elif strategy=='PertAL':
            selection_method='maxdist'
            kernel_transforms=[] 
            sel_with_train = True
            base_kernel = 'diff_effect'
            add_ctrl = False 
            
        if self.dataset_name == 'replogle_k562_essential_1000hvg':
            kernel_list=[prior_scfm_kernel]
            kernel_list.extend(['rpe1_kernel','ops_A549_kernel', 'biogpt_kernel',
                                'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel'])
        
        elif self.dataset_name =='replogle_rpe1_essential_1000hvg':
            kernel_list=[prior_scfm_kernel]
            kernel_list.extend(['k562_kernel','ops_A549_kernel', 
                                'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel'])
        
        elif self.dataset_name == 'adamson_k562_essential_1000hvg':
            kernel_list=[prior_scfm_kernel]
            # Note: RPE1 prior is not used here to avoid potential data loss,
            # as the goal is to preserve as much perturbation data as possible.
            kernel_list.extend(['ops_A549_kernel', 'ops_HeLa_HPLM_kernel', 'ops_HeLa_DMEM_kernel'])

        else:
            raise ValueError('Kernel is not preprocessed yet for this new dataset...')
        prior_kernel_list, prior_feat_list = [],[] 

        url = 'https://dataverse.harvard.edu/api/access/datafile/7377301'
        data_path = os.path.join(self.path, self.dataset_name + '_kernels')
        
        zip_data_download_wrapper(url, data_path, data_path)
        data_path = data_path + '/knowledge_kernels_1k/'
        print('Kernel in ', data_path)
        for i in kernel_list:
            _, k, f = load_kernel(data_path, i)
            prior_kernel_list.append(k)
            prior_feat_list.append(f)
        pert_list, true_gold, truth_feat = load_kernel(data_path, 'ground_truth_delta')
        pert_list = [i.split('+')[0] for i in pert_list]
        
        strategy = get_strategy('active_learning')(self.dataset, self.net, 
                                                                selection_method, base_kernel, 
                                                                kernel_transforms, self.device, 
                                                                sel_with_train, 
                                                                reduce_latent_feat_dim_via_pca = False, 
                                                                use_prior_only = False, 
                                                                integrate_mode = 'mean_new', 
                                                                normalize_mode = 'max', 
                                                                prior_kernel_list = prior_kernel_list, 
                                                                prior_kernel_pert_list = pert_list, 
                                                                train_gold = true_gold, 
                                                                add_ctrl = add_ctrl, 
                                                                prior_feat_list = prior_feat_list, 
                                                                gene_hvg_idx = None, 
                                                                lamb = 2,strategy_name=strategy,dataset_name=self.dataset_name,
                                                                llm_name=self.llm_name,
                                                                gradient_weight=self.gradient_weight )
        
        self.strategy = strategy

    def start(self, n_init_labeled = 100, n_round = 5, n_query = 100, save_kernel = False, save_path = None):
        if save_path is None:
            save_path = self.path
        round2query = {}

        # Initialize the labeled data pool with 'n_init_labeled' samples
        init_idx = self.dataset.initialize_labels(n_init_labeled)
        round2query[0] = self.dataset.pert_train[init_idx]

        # Print the number of labeled, unlabeled, and testing samples
        print(f"number of labeled pool: {n_init_labeled}")
        print(f"number of unlabeled pool: {self.dataset.n_pool-n_init_labeled}")
        print(f"number of testing pool: {self.dataset.n_test}")
        print()

        # Start the first round of training (Round 0)
        print("----- Round 0/", n_round, " ----")
        self.strategy.train()

        # Evaluate the model performance on the test data
        res, out = self.strategy.eval(self.test_data)
        metrics = ['pearson_delta', 
                    'frac_opposite_direction_top20_non_dropout',
                    'mse_non_dropout',
                    'mse_top20_de_non_dropout',
                    'pearson_delta_top20_de_non_dropout',
                    'mse_4_non_dropout', 
                    'mse_4_top20_de_non_dropout']
        
        # Log the evaluation results to wandb
        if self.wandb:
            for m in metrics:
                self.wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

        print(f"Round 0 pearson score: {np.mean([j['pearson_delta'] for i,j in out.items() if 'pearson_delta' in j])}")


        for rd in range(1, n_round+1):
            print("----- Round ",rd,"/", n_round, " ----")

            # Query the next 'n_query' samples based on the strategy
            query_idxs = self.strategy.query(n_query, save_kernel, self.exp_name + '_round' + str(rd), round = rd + 1,llm_weight=self.llm_weight)
            round2query[rd] = self.dataset.pert_train[query_idxs]

            # Print how many new perturbations (samples) are being queried
            print('Querying ' + str(len(query_idxs)) + ' new perturbations!')
           
            # Update the strategy with the newly queried samples and retrain
            self.strategy.update(query_idxs)
            self.strategy.train()

            # Evaluate the model after the update
            res, out = self.strategy.eval(self.test_data)

            if self.wandb:
                for m in metrics:
                    self.wandb.log({'test_round_' + m: np.mean([j[m] for i,j in out.items() if m in j])})

            print(f"Round {rd} pearson delta: {np.mean([j['pearson_delta'] for i,j in out.items() if 'pearson_delta' in j])}")

        # Save the round-to-query mapping to a file
        import pickle
        if not os.path.exists(save_path + '/res'):
            os.mkdir(save_path + '/res')
        with open(save_path + '/res/' + self.exp_name + '.pkl', 'wb') as f:
            pickle.dump(round2query, f)
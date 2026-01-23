import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', type=str, default="PertAL")
parser.add_argument('--device',type=int, default=1,help='which gpu to use if any (default: 0)')
parser.add_argument('--prior_scfm_kernel',type=str, default='scgpt_blood',help='which scFM feature')
parser.add_argument('--seed',type=int, default=5,help='which experimental dataset split')
parser.add_argument('--dataset_name',type=str, default='replogle_k562',help='which experimental dataset')
parser.add_argument('--llm_name',type=str, default='gpt41-mini',help='which llm prior')
parser.add_argument('--llm_weight',type=float, default=0.2,help='Weight of the LLM prior')

args = parser.parse_args()


from pertal.pertal import PertAL
strategy = args.strategy
device = "cuda:" + str(args.device)
prior_scfm_kernel=args.prior_scfm_kernel
seed=args.seed
dataset_name=args.dataset_name
llm_name=args.llm_name
llm_weight=args.llm_weight

# Initialize wandb experiment if weight_bias_track is True
# If wandb is not installed, please set weight_bias_track to False
interface = PertAL(weight_bias_track =True, 
                     exp_name = f"{strategy}_{llm_name}_{dataset_name}_seed{seed}",
                     device = device, 
                     seed = seed,
                     llm_weight=llm_weight
                     )


path = './data/'
interface.initialize_data(path = path,
                          dataset_name=f'{dataset_name}_essential_1000hvg',
                          batch_size = 256,
                          llm_name=llm_name)

interface.initialize_model(epochs = 20, hidden_size = 64)
interface.initialize_active_learning_strategy(strategy = strategy,prior_scfm_kernel=prior_scfm_kernel)

interface.start(n_init_labeled =100, n_round = 5, n_query =100,save_path='./results')
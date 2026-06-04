import argparse
import importlib

def execute_function(method, mode):

    if method in ['path', 'path_ada']:
        mode = 'main'
        module_name = f"models.{method}.main"
        print(module_name)

        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, 'main')

    elif method == 'imagentime':
        module_name = f"models.imagentime.{mode}"
        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, 'main')

    elif method == 'sbts':
        mode = 'main'
        module_name = f"models.{method}.main"
        print(module_name)

        train_module = importlib.import_module(module_name)
        train_function = getattr(train_module, 'main')

    else:
        print('HERE')
        module_name = f"models.{method}.{mode}"

        try:
            train_module = importlib.import_module(module_name)
            train_function = getattr(train_module, 'main')
        except ModuleNotFoundError as e:
            print(f"Module {module_name} not found. Cause: {e}")
            exit(1)
        except AttributeError:
            print(f"Function 'main' not found in module {module_name}.")
            exit(1)
    return train_function

def get_args():
    parser = argparse.ArgumentParser(description='Pipeline')

    # General configs
    parser.add_argument('--dataname', type=str, default='sine', help='Name of dataset.')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or sample.')
    parser.add_argument('--method', type=str, default='path', help='Method: path or baseline.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    #parser.add_argument('--mode', type=str, default='train')


    # PATH arguments
    parser.add_argument('--K', type=int, default=1, help='Markov Order')
    parser.add_argument('--kernel',type=int, choices=[0,1,2,3], default=0 ,help="QUARTIC = 0; GAUSSIAN = 1; EPANECHNIKOV = 2; TRIANGULAR = 3;")
    parser.add_argument('--h',type=float, default=0.05, help='bandwith')
    parser.add_argument('--q',type=float, default=0.001,help='neighbour fraction')

    # SBTS arguments
    parser.add_argument('--N_pi', type=int, default=100)

    # TimeGAN

    parser.add_argument('--z_dim',help='z or data dimension',default=6,type=int)
    parser.add_argument('--seq_len',help='sequence length',default=24,type=int)
    parser.add_argument('--module',choices=['gru', 'lstm', 'lstmLN'],default='gru',type=str)
    parser.add_argument('--hidden_dim',help='hidden state dimensions (should be optimized)',default=24,type=int)
    parser.add_argument('--num_layer',help='number of layers (should be optimized)',default=3,type=int)
    parser.add_argument('--iteration',help='Training iterations (should be optimized)',default=50000,type=int)
    parser.add_argument('--batch_size',help='the number of samples in mini-batch (should be optimized)',default=128,type=int)
    parser.add_argument('--metric_iteration',help='iterations of the metric computation',default=10,type=int)
    parser.add_argument('--print_freq', type=int, default=1000, help='frequency of showing training results on console')
    parser.add_argument('--load_weights', action='store_true', help='Load the pretrained weights')
    parser.add_argument('--resume', default='', help="path to checkpoints (to continue training)")

    parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')

    parser.add_argument('--w_gamma', type=float, default=1, help='Gamma weight')
    parser.add_argument('--w_es', type=float, default=0.1, help='Encoder loss weight')
    parser.add_argument('--w_e0', type=float, default=10, help='Encoder loss weight')
    parser.add_argument('--w_g', type=float, default=100, help='Generator loss weight.')


    # TSDIFF (CSPD;DSPD)
    #parser.add_argument('--seed', type=int, default=1)
    #parser.add_argument('--dataset', type=str, choices=['cir', 'lorenz', 'ou', 'predator_prey', 'sine', 'sink'])
    parser.add_argument('--diffusion', type=str, choices=[
        'GaussianDiffusion', 'OUDiffusion', 'GPDiffusion',
        'ContinuousGaussianDiffusion', 'ContinuousOUDiffusion', 'ContinuousGPDiffusion',
    ])
    parser.add_argument('--model', default='rnn', type=str, choices=['feedforward', 'rnn', 'cnn', 'ode', 'transformer'])
    parser.add_argument('--gp_sigma', type=float, default=0.1)
    parser.add_argument('--ou_theta', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (tsdiff).')
    parser.add_argument("--discrete_num_steps", type=int, default=1000)


    # configs for sampling
    parser.add_argument('--save_path', type=str, default=None, help='Path to save synthetic data.')
    parser.add_argument('--steps', type=int, default=50, help='NFEs.')
    
    args = parser.parse_args()

    return args
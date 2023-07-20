import argparse

def base_train_args():
    parser = argparse.ArgumentParser(add_help=False)
    # mode
    parser.add_argument('--mode', type=str, default="train", choices=['train', 'eval'],
                        help="train or eval")
    parser.add_argument('--model_path', type=str, default="#",
                        help="existing model path")
    parser.add_argument('--debug', action='store_true',default=False,
                        help='for backup')
    parser.add_argument('--random', action='store_true',default=True,
                        help='for backup')
    parser.add_argument('--warm', action='store_true', default=True,
                        help='warm-up for large batch training')
    parser.add_argument('--fix', action='store_true', default=False,
                        help='fix')
    parser.add_argument('--cosine', action='store_true', default=True,
                        help='using cosine annealing')

    # optimization
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'sgd'],
                        help='optimizer.')
    parser.add_argument('--criterion', type=str, default='bce', choices=['bce', 'conloss'],
                        help='criterion.')
    parser.add_argument('--weight_decay', type=float, default=0,  # 1e-6
                        help='decrease overfitting.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,350,500',
                        help='where to decay lr, can be a list')


    # training configuration
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs. start from 1')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--print_freq', type=int, default=20,
                        help='print frequency')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early-stopping.')
    parser.add_argument('--gpu', type=str, default="0,1,2",
                        help='GPU ids')

    return parser

def database_args():
    parser = argparse.ArgumentParser(add_help=False)

    # dataset
    parser.add_argument('--dataset', type=str, default='MIMIC3', choices=['physionet2012', 'MIMIC3', 'MIMIC4'],
                        help="dataset")
    parser.add_argument('--application', type=str, default='inhos_mortality',
                        choices=['inhos_mortality', 'length_of_stay_3', 'length_of_stay_7', "length_of_stay_multi"],
                        help="dataset")
    parser.add_argument('--folds', type=list, default=[3],
                        help='folds id')
    parser.add_argument('--ffill', action='store_true',default=True,
                        help='data filling， ffill or None')
    parser.add_argument('--standardization', action='store_false',default=True,
                        help='standardization for the training dataset')
    parser.add_argument('--data_clip', action='store', default=False,
                        help='data clipping: decide the maximun and minimun value of the training dataset')
    parser.add_argument('--data_clip_min', type=float, default=-1*float('inf'),
                        help='data clipping: minimun value of the training dataset')
    parser.add_argument('--data_clip_max', type=float, default=float('inf'),
                        help='data clipping: maximun value of the training dataset')
    return parser

def regular_dataset():
    regular_parser = argparse.ArgumentParser(description="regular model.", add_help=False,
                                             parents=[database_args()])
    regular_parser.add_argument('--dataset_mode', type=str, default='regular', choices=['regular'],
                        help="regular or irregular")
    regular_parser.add_argument('--ffill_steps', type=int, default=48,
                        help='data filling steps')
    regular_parser.add_argument('--max_timesteps', type=int, default=48,
                        help='Time series of at most # time steps are used. Default: 48.')
    return regular_parser


def CohortNet_parse_args():
    parser = argparse.ArgumentParser(description="Run CohortNet on regular dataset.", add_help=False)

    parser.add_argument('--model', type=str, default='CohortNet')
    # for mimic
    parser.add_argument('--embed_dim', type=int, default=24,
                        help="embed_dim")
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help="hidden_dim")
    parser.add_argument('--fusion_dim', type=int, default=32,
                        help='fusion_dim for interaction features')
    parser.add_argument('--compress_dim', type=int, default=24,
                        help='compress_dim for features')

    parser.add_argument('--max_cohort_size', type=int, default=8000,
                        help='maximal number for cohorts')
    parser.add_argument('--topn', type=int, default=2,
                        help='N for topn features')
    parser.add_argument('--k', type=int, default=7,
                        help='k for Kmeans')
    parser.add_argument('--min_freq', type=int, default=10,
                        help='minimal frequency ')
    parser.add_argument('--min_sample_freq', type=int, default=5,
                        help='minimal frequency on samples')

    parser.add_argument('--clip_min', type=float, default=-3.0,
                        help="clip_min")
    parser.add_argument('--clip_max', type=float, default=3.0,
                        help="clip_max")
    parser.add_argument('--inter_type', type=str, default="mul", choices=['mul', 'add'],
                        help='interaction types')
    parser.add_argument('--active', type=str, default="relu", choices=['dropout', 'relu'],
                        help='activation types')

    return parser

def Cohort_args():
    cohort_args = argparse.ArgumentParser(description="cohort learning mode.", add_help=False)
    cohort_args.add_argument('--cohort_epoch', type=int, default=0,
                             help="the epoch to start calculate cohort")
    cohort_args.add_argument('--cohort_iter', type=int, default=-1,
                             help="the iteration epoch to recalculate cohort")
    return cohort_args

def get_parser():
    base_parser = argparse.ArgumentParser(description="Healthcare Model.",add_help=False)

    subparsers = base_parser.add_subparsers(help='commands')

    subparsers.add_parser('CohortNet', help='CohortNet',
                          parents=[base_train_args(), regular_dataset(), CohortNet_parse_args(), Cohort_args()])

    return base_parser.parse_args()
import argparse

def parser():
    parser = argparse.ArgumentParser(description='DLCV_final')
    parser.add_argument('--todo', choices=['train', 'valid', 'test', 'visualize'], default='train',
        help='what behavior want to do: train | valid | test')

    parser.add_argument('--dataset', default='cifar100', help='the dataset to train')
    parser.add_argument('--model_type', default='gnn', help='model to use')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--pretrain_dir', default='')
    parser.add_argument('--use_gpu', default='0', help='which use want to use')
    parser.add_argument('--seed', default=1, type=int, help='the seed for noise')
    parser.add_argument('--batch_size', default=16, type=int, help='size of data per training iteration')
    parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--max_iteration', default=100000, type=int, help='max iteration to train')
    parser.add_argument('--log_interval', default=100, type=int)
    parser.add_argument('--eval_interval', default=2000, type=int)
    parser.add_argument('--early_stop', default=5, type=int, 
        help='the number of epochs to stop training if the loss is not decrease')
    parser.add_argument('--early_stop_pretrain', default=5, type=int, help='early stop for pretrain')
    parser.add_argument('--test_dir', default='')
    parser.add_argument('--data_root', default='data', help='root for train data')
    parser.add_argument('--log_root', default='log', help='the root to save log')
    parser.add_argument('--model_root', default='model', help='the root to save model')
    parser.add_argument('--affix', default='', help='affix for the name of save folder')
    parser.add_argument('--save', action='store_true', help='whether to save model and logs')
    parser.add_argument('--load', action='store_true', help='whether to load model')
    parser.add_argument('--load_dir', default='', help='the model to load')
    parser.add_argument('--output_dir', default='output', help='the folder to save output')
    parser.add_argument('--output_name', default='output.txt', help='filename of output')
    parser.add_argument('--nway', default=20, type=int)
    parser.add_argument('--shots', default=5, type=int)
    parser.add_argument('--freeze_cnn', action='store_true', help='whether to freeze cnn-embedding layer')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))
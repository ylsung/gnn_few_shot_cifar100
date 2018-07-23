from trainer import Trainer
from data import self_Dataset, self_DataLoader

import os
import time
import random
import numpy as np
from argument import print_args, parser
from utils import create_logger, mkdir

def main(args):
    if args.todo == 'train' or args.todo == 'valid':
        folder_name = '%dway_%dshot_%s_%s' % (args.nway, args.shots, args.model_type, args.affix)
        model_folder = os.path.join(args.model_root, folder_name)
        log_folder = os.path.join(args.log_root, folder_name)

        mkdir(args.model_root)
        mkdir(args.log_root)
        mkdir(model_folder)
        mkdir(log_folder)
        setattr(args, 'model_folder', model_folder)
        setattr(args, 'log_folder', log_folder)
        logger = create_logger(log_folder, args.todo)
        print_args(args, logger)

        tr_dataloader = self_DataLoader(args.data_root, 
            train=True, dataset=args.dataset, seed=args.seed, nway=args.nway)

        trainer_dict = {'args': args, 'logger': logger, 'tr_dataloader': tr_dataloader}

        trainer = Trainer(trainer_dict)

        ###########################################
        ## pretrain CNN embedding

        if args.pretrain:
            if args.pretrain_dir != '':
                pretrain_path = os.path.join(args.pretrain_dir, 'pretrain_model.pth')
                trainer.load_pretrain(pretrain_path)
            else:
                pretr_tr_data, pretr_tr_label = tr_dataloader.get_full_data_list() # already shuffled the data

                va_size = int(0.1 * len(pretr_tr_data))

                pretr_tr_dataset = self_Dataset(pretr_tr_data[va_size:], pretr_tr_label[va_size:])
                pretr_va_dataset = self_Dataset(pretr_tr_data[:va_size], pretr_tr_label[:va_size])

                logger.info('start pretraining...')

                trainer.pretrain(pretr_tr_dataset, pretr_va_dataset)

                logger.info('finish pretraining...')

        ###########################################
        ## load the model trained before

        if args.load:
            model_path = os.path.join(args.load_dir, 'model.pth')
            trainer.load_model(model_path)

        ###########################################
        ## start training

        trainer.train()

    elif args.todo == 'test':

        print(args.load_dir)
        
        logger = create_logger('', args.todo)
        print_args(args, logger)    

        te_dataloader = self_DataLoader(args.data_root, 
            train=False, dataset=args.dataset, seed=args.seed, nway=args.nway)

        trainer_dict = {'args': args, 'logger': logger}

        trainer = Trainer(trainer_dict)

        test_data_list, test_label_list = te_dataloader.get_few_data_list()

        test_data_array, test_label_array = np.stack(test_data_list), np.hstack(test_label_list)

        if args.load:
            model_path = os.path.join(args.load_dir, 'model.pth')
            trainer.load_model(model_path)

        test_pred = trainer.test(test_data_array, te_dataloader)

        print(test_pred.shape, test_label_array.shape)

        correct = (test_pred == test_label_array).sum()
        test_acc = (test_pred == test_label_array).mean() * 100.0

        print('test_acc: %.4f %%, correct: %d / %d' % (test_acc, correct, len(test_label_array)))

if __name__ == '__main__':
    args = parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.use_gpu
    main(args)
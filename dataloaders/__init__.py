from .rif import RIF
import os
from torch.utils.data import DataLoader
import logging
import torch



def build_dataloader(args, clients):

    train_dls = []
    val_dls = []
    test_dls = []

    dataset_lens = []

    for idx, client in enumerate(clients):
        train_set = RIF(client_idx=idx, base_path=args.data_root,
                             split='train', transform=None, isVal=0)
        valid_set = RIF(client_idx=idx, base_path=args.data_root,
                             split='train', transform=None, isVal=1)
        test_set = RIF(client_idx=idx, base_path=args.data_root,
                             split='test', transform=None)
 
        logging.info('{} train  dataset: {}'.format(client, len(train_set)))
        logging.info('{} val  dataset: {}'.format(client, len(valid_set)))
        logging.info('{} test  dataset: {}'.format(client, len(test_set)))
 

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                               shuffle=True, drop_last=True)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=args.batch_size,
                                               shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
                                              shuffle=False, drop_last=False)

        train_dls.append(train_loader)
        val_dls.append(valid_loader)
        test_dls.append(test_loader)

        dataset_lens.append(len(train_set))
    
    client_weight = []
    total_len = sum(dataset_lens)
    for i in dataset_lens:
        client_weight.append(i / total_len)

    return train_dls, val_dls, test_dls, client_weight

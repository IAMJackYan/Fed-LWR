import logging
import copy
from .unet import UNet,UNet_pro

def build_model(args, clients, device):

    n_classes = 2

    if args.dataset=='pmri':
        n_classes = 3 

    if args.model == 'unet':
        model = UNet(out_channels=n_classes)
    elif args.model == 'unet_pro':
        model = UNet_pro(out_channels=n_classes)
    else:
        logging.info('unknow model')
        return

    model = model.to(device)

    local_models = []
    for id, c in enumerate(clients):
        

        local_models.append(copy.deepcopy(model))

        total_params = sum(p.numel() for p in model.parameters())
        logging.info('{}: model parameters {} M'.format(c, total_params/1024/1024))

    global_model = copy.deepcopy(model)
    return local_models, global_model

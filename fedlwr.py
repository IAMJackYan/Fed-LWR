import logging
import torch
import os
import numpy as np
import random
import argparse

from pathlib import Path
from utils import set_for_logger
from dataloaders import build_dataloader

from loss import DiceLoss
import torch.nn.functional as F
from nets import build_model

from loss import DiceLoss
import copy
from cka import CKACalculator

@torch.no_grad()
def cal_contribution_of_clients(global_model, local_models, train_loaders, device, layer_names):

    for model in local_models:
        model.eval()

    global_model.eval()

    lay_wise_weight = {}

    for cid, loader in enumerate(train_loaders):
        calculator = CKACalculator(model1=global_model, model2=local_models[cid], dataloader=loader, hook_fn='flatten', num_epochs=5)
        cka_output = calculator.calculate_cka_matrix()

        lay_index = cka_output.shape[0]

        for i in range(lay_index):
            if i not in lay_wise_weight.keys():
                lay_wise_weight[i] = [cka_output[i][i].cpu().item()]
            else:
                lay_wise_weight[i].append(cka_output[i][i].cpu().item())
        
        calculator.reset()
        torch.cuda.empty_cache()
    
    lay_wise_weight_map = {}
    for i in range(lay_index):
        lay_wise_weight[i] = np.clip(lay_wise_weight[i], a_min=1e-4, a_max=0.99)
        lay_wise_weight[i] = 1 - lay_wise_weight[i]
        lay_wise_weight[i] /=  (np.sum(lay_wise_weight[i]) + 1e-9)
        lay_wise_weight_map[layer_names[i]] = lay_wise_weight[i]

    return lay_wise_weight_map      


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help="Random seed")
    parser.add_argument('--data_root', type=str, required=False, default="", help="Data directory")
    parser.add_argument('--dataset', type=str, default='pmri')
    parser.add_argument('--model', type=str, default='unet')

    parser.add_argument('--rounds', type=int, default=200, help='number of maximum communication round')
    parser.add_argument('--epochs', type=int, default=1, help='number of local epochs')
    parser.add_argument('--device', type=str, default='cuda:3', help='The device to run the program')

    parser.add_argument('--log_dir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument('--save_dir', type=str, required=False, default="./weights/", help='Log directory path')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="L2 regularization strength")
    parser.add_argument('--batch-size', type=int, default=8, help='input batch size for training (default: 64)')
    parser.add_argument('--experiment', type=str, default='experiment', help='The device to run the program')

    parser.add_argument('--test_step', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.6, help="")
    parser.add_argument('--cka_type', type=str, default='linear')

    args = parser.parse_args()
    return args

def init_model_weight(model, weight):
    for key in weight.keys():
        if key not in model.state_dict().keys():
            pass
        else:
            model.state_dict()[key].data.copy_(weight[key])


def communication(server_model, models, client_weights):
    with torch.no_grad():
        for key in server_model.state_dict().keys():
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models



def communication_ours(server_model, models, lay_wise_client_weights):
    with torch.no_grad():
        for lay_id, key in enumerate(server_model.state_dict().keys()):

            t = key.split('.')
            if len(t) > 2:
                l_name = t[0] + '.' + t[1]
            else:
                l_name = t[0]
            
            client_weights = lay_wise_client_weights[l_name]
            temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)

            for client_idx in range(len(client_weights)):
                temp += client_weights[client_idx] * models[client_idx].state_dict()[key]

            server_model.state_dict()[key].data.copy_(temp)
            for client_idx in range(len(client_weights)):
                models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models


def train(cid, model, dataloader, device, optimizer, epochs, loss_func):
    model.train()

    for epoch in range(epochs):
        train_acc = 0.
        loss_all = 0.
        for x, target in dataloader:

            x = x.to(device)

            target = target.to(device)

            output = model(x)
            
            optimizer.zero_grad()

            loss = loss_func(output, target)
            loss_all += loss.item()

            train_acc += DiceLoss().dice_coef(output, target).item()

            loss.backward()
            optimizer.step()

        avg_loss = loss_all / len(dataloader)
        train_acc = train_acc / len(dataloader)
        logging.info('Client: [%d]  Epoch: [%d]  train_loss: %f train_acc: %f'%(cid, epoch, avg_loss, train_acc))

def get_model_layer_name(model):
    layer_names = []
    for name in model.state_dict().keys():
        t = name.split('.')
        if len(t) > 2:
            l_name = t[0] + '.' + t[1]
        else:
            l_name = t[0]
        if l_name not in layer_names:
            layer_names.append(l_name)
    return layer_names


def test(model, dataloader, device, loss_func):
    model.eval()

    loss_all = 0
    test_acc = 0

    with torch.no_grad():
        for x, target in dataloader:

            x = x.to(device)
            target = target.to(device)

            output = model(x)
            loss = loss_func(output, target)
            loss_all += loss.item()

            test_acc += DiceLoss().dice_coef(output, target).item()
        

    acc = test_acc/ len(dataloader)
    loss = loss_all / len(dataloader)

    return loss, acc


def main(args):
    set_for_logger(args)
    logging.info(args)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device(args.device)

    clients = ['site1', 'site2', 'site3', 'site4']

    # build dataset
    train_dls, val_dls, test_dls, client_weight = build_dataloader(args, clients)

    # build model
    local_models, global_model = build_model(args, clients, device)

    layer_names = get_model_layer_name(global_model)

    # build loss
    loss_fun = DiceLoss()

    optimizer = []
    for id in range(len(clients)):
        optimizer.append(torch.optim.Adam(local_models[id].parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)))

    best_dice = 0
    best_dice_round = 0
    best_local_dice = []

    weight_save_dir = os.path.join(args.save_dir, args.experiment)
    Path(weight_save_dir).mkdir(parents=True, exist_ok=True)
    logging.info('checkpoint will be saved at {}'.format(weight_save_dir))

    client_weight = []
    for idx, client in enumerate(clients):
        client_weight.append(1/len(clients))
    
    
    for r in range(args.rounds):

        logging.info('-------- Commnication Round: %3d --------'%r)

        global_w = global_model.state_dict()

        for idx, client in enumerate(clients):
            train(idx, local_models[idx], train_dls[idx], device, optimizer[idx], args.epochs, loss_fun)
            

        temp_locals = copy.deepcopy(local_models)

        #commnication
        communication(global_model, local_models, client_weight)

        lay_wise_weights = cal_contribution_of_clients(global_model, temp_locals, val_dls, device,  layer_names)

        communication_ours(global_model, temp_locals, lay_wise_weights)

        global_w = global_model.state_dict()
        for idx, client in enumerate(clients):
            local_models[idx].load_state_dict(global_w)


        if r% args.test_step == 0:
            #test
            avg_loss = []
            avg_dice = []
            for idx, client in enumerate(clients):
                loss, dice = test(local_models[idx], test_dls[idx], device, loss_fun)

                logging.info('client: %s  test_loss:  %f   test_acc:  %f '%(client, loss, dice))
                avg_dice.append(dice)
                avg_loss.append(loss)

            avg_dice_v = sum(avg_dice) / len(avg_dice)
            avg_loss_v = sum(avg_loss) / len(avg_loss)

            logging.info('Round: [%d]  avg_test_loss: %f avg_test_acc: %f std_test_acc: %f'%(r, avg_loss_v, avg_dice_v, np.std(np.array(avg_dice))))

            if best_dice < avg_dice_v:
                best_dice = avg_dice_v
                best_dice_round = r
                best_local_dice = avg_dice

                weight_save_path = os.path.join(weight_save_dir, 'best.pth')
                torch.save(global_model.state_dict(), weight_save_path)
            

    logging.info('-------- Training complete --------')
    logging.info('Best avg dice score %f at round %d '%( best_dice, best_dice_round))
    for idx, client in enumerate(clients):
        logging.info('client: %s  test_acc:  %f '%(client, best_local_dice[idx]))


if __name__ == '__main__':
    args = get_args()
    main(args)





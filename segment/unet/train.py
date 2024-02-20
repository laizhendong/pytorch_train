import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet


from utils.dataset import JiangsiDataset
from torch.utils.data import DataLoader

from utils.calc_class_weights import calc_class_weights

from config import CFG


def train_net(net,dataset_train, dataset_test,device):

    train_loader = DataLoader(dataset_train, batch_size=CFG.data_params['batch_size'],
                              shuffle=True, num_workers=CFG.threads, pin_memory=True)
    val_loader = DataLoader(dataset_test,batch_size=CFG.data_params['batch_size'],
                            shuffle=False, num_workers=CFG.threads, pin_memory=True, drop_last=True)


    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {CFG.epochs}
        Batch size:      {CFG.data_params['batch_size']}
        Learning rate:   {CFG.lr_params['base_lr']}
        Training size:   {len(dataset_train)}
        Validation size: {len(dataset_test)}
        Device:          {device.type}
        input_size:      {CFG.augment_params['resize'][0]} x {CFG.augment_params['resize'][1]} 
    ''')

    #optimizer = optim.RMSprop(net.parameters(), lr=CFG.lr_params['base_lr'], weight_decay=1e-8, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=CFG.lr_params['base_lr'], weight_decay=1e-8)

    scheduler = None
    if CFG.lr_params['policy'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    elif CFG.lr_params['policy'] == 'cosine':
        max_iter = (CFG.epochs * len(dataset_train) + CFG.data_params['batch_size'] - 1)//CFG.data_params['batch_size']
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=max_iter,eta_min=1e-5,last_epoch=-1)
    else:
        policy = CFG.lr_params['policy']
        logging.info(f'unk lr policy: {policy}')
        sys.exit(0)

    if CFG.data_params['auto_class_weight']:
        class_weights = calc_class_weights(dataset_train.class_size(),dataset_train.class_present())
        class_weights = torch.from_numpy(np.asarray(class_weights)).type(torch.FloatTensor).to(device)
    else:
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)


    for epoch in range(CFG.epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=len(dataset_train), desc=f'Epoch {epoch + 1}/{CFG.epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, torch.squeeze(true_masks).long())
                epoch_loss += loss.item()


                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                if CFG.clip_value > 0:
                    nn.utils.clip_grad_value_(net.parameters(), CFG.clip_value)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1

                if CFG.lr_params['policy'] != 'plateau':
                    scheduler.step(global_step)

                if global_step % CFG.valid_freq == 0:
                    val_score = eval_net(net, val_loader, device)
                    if CFG.lr_params['policy'] != 'plateau':
                        scheduler.step(val_score)
                    torch.save(net.state_dict(),os.path.join(CFG.saved_dir,  f'CP_epoch{epoch + 1}_{val_score}.pth'))
        logging.info(f'Epoch Loss {epoch_loss} lr {scheduler.get_last_lr()}')
    torch.save(net.state_dict(),os.path.join(CFG.saved_dir,  f'CP_final.pth'))
    logging.info(f'Checkpoint {epoch + 1} saved !')


if __name__ == '__main__':
    CFG.build_output()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    dataset_train = JiangsiDataset(list_file=CFG.datalist['train'],train_flag=True)
    dataset_test = JiangsiDataset(list_file=CFG.datalist['test'], train_flag=False)



    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=dataset_train.class_num(), bilinear=CFG.network['bilinear'])
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if not CFG.network['pretrained'] is None:
        net.load_state_dict(torch.load(CFG.network['pretrained']))


    ###todo: loading pretrained
    net.to(device=device)

    try:
        train_net(net,dataset_train, dataset_test,device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

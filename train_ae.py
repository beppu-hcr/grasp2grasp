import os
import time
import torch
import argparse
import json
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from dataset.mgg_dataset import GraspDataset
# from network.autoencoder.autoencoder import Autoencoder
import numpy as np
import random
from utils import utils_loss
from utils.utils import  makepath, makelogger , convert_euler_to_rotmat
from utils.loss import CVAE_loss_mano, CMap_loss, CMap_loss1, CMap_loss3, CMap_loss4, inter_penetr_loss, CMap_consistency_loss, set_random_seed,kl_div_normal
from pytorch3d.loss import chamfer_distance
# import mano
import ipdb
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# from evaluation.vis import vis_hand
# from manopth.manolayer import grabManoLayer
# from manotorch.manolayer import ManoLayer, MANOOutput
from datetime import datetime
from grippers.HandLayer import grabHandLayer

# TODO: Need to pass config into this
# hand_layer = grabHandLayer(ncomps=45, flat_hand_mean=True, side="right", mano_root=os.path.join("assets/mano_v1_2/models"), use_pca=False, joint_rot_mode="rotmat").to("cuda")

def train(args, writer, val_writer, epoch, model, train_loader, val_loader, device, optimizer, 
          logger, checkpoint_root, best_train_loss, best_val_loss, hand_layer, val_freq=0.125):
    since = time.time()
    logs = defaultdict(list)
    a, b, c, d, e , f= args.weight
    model.train()
    for batch_idx, input in tqdm(enumerate(train_loader), total=len(train_loader)): 
        obj_pc = input["object_pc"].float().to(device)
        hand_pose = input['hand_pose'].float().to(device)
        hand_dofs = input['hand_dofs'].float().to(device)
        hand_pc = input['hand_pc'].float().to(device)

        with torch.no_grad():
            gt_hand_xyz = hand_layer(hand_pose, hand_dofs)
            gt_hand_param = hand_layer.get_hand_param(hand_pose, hand_dofs)
            
        optimizer.zero_grad()

        '''encoder'''
        hand_feature, _, _ = model.hand_encoder(hand_pc.permute(0,2,1))
        # pointnet
        z = model.encoder(hand_feature)

        recon_param = model.decoder(z)
        recon_xyz = hand_layer(recon_param[:, :9], recon_param[:, 9:])

        # obj xyz NN dist and idx
        obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc, gt_hand_xyz)
        obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc, recon_xyz)
        # mano param loss
        param_loss = F.mse_loss(recon_param, gt_hand_param, reduction='none').sum() / recon_param.size(0)
        # mano recon xyz loss, KLD loss
        # recon_loss_num, _ = chamfer_distance(recon_xyz, gt_hand_xyz, point_reduction='sum', batch_reduction='mean')
        recon_loss_num = F.mse_loss(recon_xyz, gt_hand_xyz, reduction='none').sum() / recon_xyz.size(0)

        #cmap_loss = CMap_loss(obj_pc.permute(0,2,1)[:,:,:3], recon_xyz, obj_cmap)
        # if epoch >= 2:
        #     cmap_loss = CMap_loss3(obj_pc[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
        # else:
        #     cmap_loss = torch.tensor(0.0).to(device)
        cmap_loss = torch.tensor(0.0).to(device)
        # cmap consistency loss
        consistency_loss = CMap_consistency_loss(obj_pc[:,:,:3], recon_xyz, gt_hand_xyz,
                                                 obj_nn_dist_recon, obj_nn_dist_gt)
        # inter penetration loss
        hand_face = hand_layer.concatenated_faces
        hand_faces = hand_face.expand(recon_xyz.shape[0], -1, -1) # align the B
        if d == 0:
            penetr_loss = torch.tensor(0.0).to(device)
        else:
            penetr_loss = inter_penetr_loss(recon_xyz, hand_faces, obj_pc[:,:,:3],
                                            obj_nn_dist_recon, obj_nn_idx_recon)

        if f == 0:
            kl_loss = torch.tensor(0.0).to(device)
        else:
            kl_loss = kl_div_normal(z)


        if epoch >= 0:
            loss = a * recon_loss_num + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
        else:
            loss = a * recon_loss_num + b * param_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
        loss.backward()
        optimizer.step()
        
        logs['recon_loss'].append(recon_loss_num)
        logs['loss'].append(loss.item())
        logs['param_loss'].append(param_loss.item())
        logs['cmap_loss'].append(cmap_loss.item())
        logs['penetr_loss'].append(penetr_loss.item())
        logs['cmap_consistency'].append(consistency_loss.item())
        logs['kl_loss'].append(kl_loss.item())

        writer.add_scalar('loss', loss.item() , batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('recon_loss', recon_loss_num, batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('param_loss', param_loss.item(), batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('cmap_loss', cmap_loss.item(), batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('penetr_loss', penetr_loss.item(), batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('cmap_consistency', consistency_loss.item(), batch_idx + (epoch-1) * len(train_loader))
        writer.add_scalar('kl_loss', kl_loss.item(), batch_idx + (epoch-1) * len(train_loader))

        if batch_idx % 10 == 0:
            out_str = "Epoch: {:02d}/{:02d}, Batch: {:04d}/{:04d}, Mean Total Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f} , KL {:9.5f}".format(
                epoch, args.epochs, batch_idx, len(train_loader),
                loss.item(),
                recon_loss_num,
                param_loss.item(),
                cmap_loss.item(),
                consistency_loss.item(),
                penetr_loss.item(),
                kl_loss.item()
            )
            logger(out_str)
        
        if batch_idx % int(len(train_loader) * val_freq) == 0 and batch_idx != 0:
            val_loss = val(args, val_writer, epoch, batch_idx, len(train_loader), model, val_loader,
                           device, logger, checkpoint_root, best_val_loss, hand_layer, mode='test') #mode='val'
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save_name = os.path.join(checkpoint_root, f'model_best_val.pth')
                save_name = os.path.join(checkpoint_root, f'model_best_test.pth')
                torch.save({
                    'network': model.state_dict(),
                    'epoch': batch_idx + (epoch-1) * len(train_loader)
                }, save_name)
            model.train()

    writer.add_scalar('loss_epoch', sum(logs['loss']) / len(logs['loss']) , epoch)
    writer.add_scalar('recon_loss_epoch', sum(logs['recon_loss']) / len(logs['recon_loss']) , epoch)
    writer.add_scalar('param_loss_epoch', sum(logs['param_loss']) / len(logs['param_loss']) , epoch)
    writer.add_scalar('cmap_loss_epoch', sum(logs['cmap_loss']) / len(logs['cmap_loss']) , epoch)
    writer.add_scalar('penetr_loss_epoch', sum(logs['penetr_loss']) / len(logs['penetr_loss']), epoch)
    writer.add_scalar('cmap_consistency_epoch', sum(logs['cmap_consistency']) / len(logs['cmap_consistency']) , epoch)
    writer.add_scalar('kl_loss_epoch', sum(logs['kl_loss']) / len(logs['kl_loss']) , epoch)

    mean_recon_loss = sum(logs['param_loss']) / len(logs['param_loss']) 
        
    out_str = "Epoch: {:02d}/{:02d}, train, Mean Toal Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f} , KL {:9.5f} , Best param-loss: {:9.5f}".format(
        epoch, args.epochs,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['kl_loss']) / len(logs['kl_loss']),
        min(best_train_loss, mean_recon_loss)
    )
    logger(out_str)
    # if mean_recon_loss < best_train_loss :
    #     save_name = os.path.join(checkpoint_root, 'model_best_train.pth')
    #     torch.save({
    #         'network': model.state_dict(),
    #         'epoch': epoch
    #     }, save_name)

    return min(mean_recon_loss , best_train_loss), best_val_loss

def val(args,writer, epoch, train_idx, total_batch, model, val_loader, device,
        logger, checkpoint_root, best_val_loss, hand_layer, mode='val'):
    # validation
    model.eval()
    a, b, c, d, e ,f  = args.weight
    logs = defaultdict(list)
    with torch.no_grad():
        for batch_idx, input in enumerate(val_loader):
            obj_pc = input["object_pc"].float().to(device)
            hand_pose = input['hand_pose'].float().to(device)
            hand_dofs = input['hand_dofs'].float().to(device)
            hand_pc = input['hand_pc'].float().to(device)

            with torch.no_grad():
                gt_hand_xyz = hand_layer(hand_pose, hand_dofs)
                gt_hand_param = hand_layer.get_hand_param(hand_pose, hand_dofs)

            hand_feature, _, _ = model.hand_encoder(hand_pc.permute(0,2,1))
            # pointnet
            z = model.encoder(hand_feature)

            recon_param = model.decoder(z)
            recon_xyz = hand_layer(recon_param[:, :9], recon_param[:, 9:])

            obj_nn_dist_gt, obj_nn_idx_gt = utils_loss.get_NN(obj_pc[:,:,:3], gt_hand_xyz)
            obj_nn_dist_recon, obj_nn_idx_recon = utils_loss.get_NN(obj_pc[:, :, :3], recon_xyz)

            # mano param loss
            param_loss = F.mse_loss(recon_param, gt_hand_param, reduction='none').sum() / recon_param.size(0)
            # recon_loss_num, _ = chamfer_distance(recon_xyz, gt_hand_xyz, point_reduction='sum', batch_reduction='mean')
            recon_loss_num = F.mse_loss(recon_xyz, gt_hand_xyz, reduction='none').sum() / recon_xyz.size(0)
            # cmap_loss = CMap_loss3(obj_pc[:,:,:3], recon_xyz, obj_nn_dist_recon < 0.01**2)
            cmap_loss = torch.tensor(0.0).to(device)
            consistency_loss = CMap_consistency_loss(obj_pc[:,:,:3], recon_xyz, gt_hand_xyz,
                                                    obj_nn_dist_recon, obj_nn_dist_gt)
            # inter penetration loss
            hand_face = hand_layer.concatenated_faces
            hand_faces = hand_face.expand(recon_xyz.shape[0], -1, -1) # align the B
            if d == 0:
                penetr_loss = torch.tensor(0.0).to(device)
            else:
                penetr_loss = inter_penetr_loss(recon_xyz, hand_faces, obj_pc[:,:,:3],
                                                obj_nn_dist_recon, obj_nn_idx_recon)
            if f == 0:
                kl_loss = torch.tensor(0.0).to(device)
            else:
                kl_loss = kl_div_normal(z)


            if epoch >= 0:
                loss = a * recon_loss_num + b * param_loss + c * cmap_loss + d * penetr_loss + e * consistency_loss + f * kl_loss
            else:
                loss = a * recon_loss_num + b * param_loss + d * penetr_loss + e * consistency_loss  + f * kl_loss
            
            logs['recon_loss'].append(recon_loss_num)
            logs['loss'].append(loss.item())
            logs['param_loss'].append(param_loss.item())
            logs['cmap_loss'].append(cmap_loss.item())
            logs['penetr_loss'].append(penetr_loss.item())
            logs['cmap_consistency'].append(consistency_loss.item())
            logs['kl_loss'].append(kl_loss.item())

        writer.add_scalar('loss', sum(logs['loss']) / len(logs['loss']) , train_idx + (epoch-1) * total_batch)
        writer.add_scalar('recon_loss', sum(logs['recon_loss']) / len(logs['recon_loss']) , train_idx + (epoch-1) * total_batch)
        writer.add_scalar('param_loss', sum(logs['param_loss']) / len(logs['param_loss']) , train_idx + (epoch-1) * total_batch)
        writer.add_scalar('cmap_loss', sum(logs['cmap_loss']) / len(logs['cmap_loss']) , train_idx + (epoch-1) * total_batch)
        writer.add_scalar('penetr_loss', sum(logs['penetr_loss']) / len(logs['penetr_loss']), train_idx + (epoch-1) * total_batch)
        writer.add_scalar('cmap_consistency', sum(logs['cmap_consistency']) / len(logs['cmap_consistency']) , train_idx + (epoch-1) * total_batch)
        writer.add_scalar('kl_loss', sum(logs['kl_loss']) / len(logs['kl_loss']) , train_idx + (epoch-1) * total_batch)

    val_loss = sum(logs['param_loss']) / len(logs['param_loss'])
    out_str = "Epoch: {:02d}/{:02d},  Batch: {:04d}/{:04d}, {} , Mean Toal Loss {:9.5f}, Mesh {:9.5f}, Param {:9.5f}, CMap {:9.5f}, Consistency {:9.5f}, Penetration {:9.5f} , KL {:9.5f} , Best param-loss: {:9.5f}".format(
        epoch, args.epochs, train_idx, total_batch, mode,
        sum(logs['loss']) / len(logs['loss']),
        sum(logs['recon_loss']) / len(logs['recon_loss']),
        sum(logs['param_loss']) / len(logs['param_loss']),
        sum(logs['cmap_loss']) / len(logs['cmap_loss']),
        sum(logs['cmap_consistency']) / len(logs['cmap_consistency']),
        sum(logs['penetr_loss']) / len(logs['penetr_loss']),
        sum(logs['kl_loss']) / len(logs['kl_loss']),
        min(best_val_loss, val_loss)
    )
    logger(out_str)
    if val_loss < best_val_loss:
        save_name = os.path.join(checkpoint_root, 'model_best_{}.pth'.format(mode))
        torch.save({
            'network': model.state_dict(),
            'epoch': train_idx + (epoch-1) * total_batch
        }, save_name)

    return min(best_val_loss, val_loss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.set_num_threads(args.num_threads)

    # Autoencoder will load CUDA so we need to import it here
    from network.autoencoder.autoencoder import Autoencoder

     # log file
    save_root = os.path.join('./logs', f"autoencoder/{args.dataset}/{args.file_name}" )
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    log_root = save_root + '/exp.log'
    # logger
    logger = makelogger(makepath(os.path.join(log_root), isfile=True)).info
    starttime = datetime.now().replace(microsecond=0)
    logger('Started training %s' % (starttime))
    gpu_brand = torch.cuda.get_device_name(0) if args.use_cuda else None
    if args.use_cuda and torch.cuda.is_available():
        logger('Using 1 CUDA cores [%s] for training!' % (gpu_brand))
    logger(args)
    logger(args.weight)

    # seed
    set_random_seed(args.seed)
    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    device_num = 1
    model = Autoencoder(
        args = args,
        obj_inchannel=args.obj_inchannel,
        cvae_encoder_sizes=args.encoder_layer_sizes,
        cvae_decoder_sizes=args.decoder_layer_sizes).to(device)

    # multi-gpu
    if device == torch.device("cuda"):
        device_ids = range(torch.cuda.device_count())
        if len(device_ids) > 1:
            model = torch.nn.DataParallel(model)
            device_num = len(device_ids)
            print("Using %d GPUs" % device_num)
    # dataset

    if 'Train' in args.train_mode:
        train_dataset = GraspDataset(root_dir=args.data_dir, hand=args.hand, split="train", seed=args.seed, contact=False)

        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.dataloader_workers,drop_last=True, pin_memory=True,
                                  persistent_workers=True)
    if 'Val' in args.train_mode:
        val_dataset = GraspDataset(root_dir=args.data_dir, hand=args.hand, split="val", seed=args.seed, contact=False)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers, pin_memory=True,
                                  persistent_workers=True)
    if 'Test' in args.train_mode:
        eval_dataset = GraspDataset(root_dir=args.data_dir, hand=args.hand, split="test", seed=args.seed, contact=False)
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.dataloader_workers, pin_memory=True,
                                  persistent_workers=True)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if args.hand_encoder == 'PointNetEncoder':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(args.epochs * x) for x in [0.1, 0.2, 0.3, 0.5]], gamma=0.5)
    elif args.hand_encoder == 'PVCNN':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.00001)
    train_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/train'))
    val_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/val'))
    test_writer = SummaryWriter(os.path.join(
        os.path.dirname(log_root), 'tensorboard/test'))

    # mano hand model
    hand_layer = grabHandLayer(hand_name=args.hand, device=device)
    best_train_loss = float('inf')
    best_val_loss = float('inf')
    best_eval_loss = float('inf')
    for epoch in range(1, args.epochs+1):
        if 'Train' in args.train_mode:
            # best_train_loss = train(args, train_writer, val_writer, epoch, model, train_loader, val_loader,
            #                         device, optimizer, logger, save_root, best_train_loss, best_val_loss, hand_layer)
            best_train_loss, best_eval_loss = train(args, train_writer, test_writer, epoch, model, train_loader, eval_loader,
                                    device, optimizer, logger, save_root, best_train_loss, best_val_loss, hand_layer)
            scheduler.step()

import os
import torch
import argparse
import json
import pytorch_lightning as pl
import shutil
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from network.model_config import get_model_cfg
from utils.loss import set_random_seed
from utils.utils import  makepath, makelogger 
from network.sbfm_trainer import LitGraspModel
from dataset.mgg_paired_pl_wrapper import GraspDataModule

def main(args):
    # Set random seed for reproducibility
    set_random_seed(args.seed)
    
    # Load model config
    cfg = get_model_cfg()
    save_root = os.path.join(args.output_dir, f"diffusion_ddp/{args.dataset}/{args.file_name}" )
    log_root = save_root + '/exp.log'

    # logger
    custom_logger = makelogger(makepath(os.path.join(log_root), isfile=True)).info
    starttime = datetime.now().replace(microsecond=0)
    custom_logger('Started training %s' % (starttime))
    gpu_brand = torch.cuda.get_device_name(0) if args.use_cuda else None
    if args.use_cuda and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        custom_logger("Using {} CUDA cores for training!".format(len(device_ids), gpu_brand))
    custom_logger(args)
    
    # Create Lightning model
    try:
        sigma = args.sigma
    except Exception as e:
        sigma = None
    
    if hasattr(args, 'ot_cost'):
        cfg.fm.ot_cost = args.ot_cost
    
    if hasattr(args, 'ema_start_step'):
        ema_start_step = args.ema_start_step
    else:
        ema_start_step = 10000
    if hasattr(args, 'from_checkpoint'):
        model = LitGraspModel.load_from_checkpoint(
            args.from_checkpoint,
            cfg=cfg,
            learning_rate=args.learning_rate,
            warmup_total_steps=args.warmup_total_steps,
            ema_decay=args.ema_decay,
            grad_clip=args.grad_clip,
            sigma=sigma,
            pbar=args.pbar,
            custom_logger=custom_logger,
            ema_start_step=ema_start_step
        )
    else:
        model = LitGraspModel(
            cfg=cfg,
            learning_rate=args.learning_rate,
            warmup_total_steps=args.warmup_total_steps,
            ema_decay=args.ema_decay,
            grad_clip=args.grad_clip,
            sigma=sigma,
            pbar=args.pbar,
            custom_logger=custom_logger,
            ema_start_step=ema_start_step,
        )
    
    use_gwh = cfg.fm.ot_cost == 'gwh'
    use_jac = cfg.fm.ot_cost == 'jac'
    # Create data module
    data_module = GraspDataModule(
        data_dir=args.data_dir,
        hand1=args.hand1,
        hand2=args.hand2,
        batch_size=args.batch_size,
        contact=True,
        gwh=use_gwh,
        jac=use_jac,
    )
    
    ckpt_dir = os.path.join(save_root, 'checkpoints')
    # Define callbacks for checkpointing and monitoring learning rate
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='model_epoch{epoch:05d}',
        save_top_k=-1,
        every_n_epochs=args.ckpt_save_interval
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    logger_dir = os.path.join(args.output_dir, f"diffusion_ddp/{args.dataset}")
    logger = TensorBoardLogger(logger_dir, name=args.file_name)
    # Create Trainer with DDP (DistributedDataParallel)
    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            super().configure_ddp()
            self.model._set_static_graph()

    all_gpus = list(range(torch.cuda.device_count()))
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",  # Use GPU accelerator
        devices=all_gpus,         # Use all available GPUs
        num_nodes=1,
        strategy=CustomDDPStrategy(),     # Use Distributed Data Parallel strategy
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=args.grad_clip,
        logger=logger,
        detect_anomaly=True
    )
    
    # Start training
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,default = "config/XXX.json",help='Path to the JSON config file')
    args = parser.parse_args()

    with open(args.config, 'r') as configfile:
        config = json.load(configfile)
    for key, value in config.items():
        parser.add_argument(f'--{key}', type=type(value), default=value)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # torch.set_num_threads(args.num_threads)

    # log file
    save_root = os.path.join(args.output_dir, f"diffusion_ddp/{args.dataset}/{args.file_name}" )
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    ckpt_dir = os.path.join(save_root, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    shutil.copy(args.config, save_root)

    main(args)

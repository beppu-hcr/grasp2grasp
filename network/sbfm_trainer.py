import copy
import torch
import pytorch_lightning as pl
from torch.optim import Adam
from utils.loss import ema, warmup_lr  # assuming these are available from your code
from network.unet.unet import UViTContact
from network.fm.cfm import SchrodingerBridgeConditionalFlowMatcher
from pytorch3d.ops import sample_farthest_points

class LitGraspModel(pl.LightningModule):
    def __init__(self, cfg, learning_rate, warmup_total_steps, ema_decay, grad_clip,
                 sigma=None, pbar=False, custom_logger=None, ema_start_step=10000):
        """
        Args:
            cfg: model configuration (assumed to contain `cfg.unet` and `cfg.fm` attributes).
            learning_rate: learning rate for the Adam optimizer.
            warmup_total_steps: total steps for the learning rate warmup.
            ema_decay: decay rate for exponential moving average.
            grad_clip: gradient clipping value.
        """
        super().__init__()
        self.save_hyperparameters(ignore=['cfg', 'pbar', 'custom_logger', 'sigma'])
        self.cfg = cfg
        self.pbar = pbar
        self.custom_logger = custom_logger
        
        # Build networks
        self.model = UViTContact(cfg.unet)
        self.score_model = UViTContact(cfg.unet)
        # Copy for EMA updates
        self.ema_model = copy.deepcopy(self.model)
        self.ema_score_model = copy.deepcopy(self.score_model)
        
        # Flow matcher
        if sigma is not None:
            self.fm = SchrodingerBridgeConditionalFlowMatcher(
                sigma=sigma, ot_method=cfg.fm.ot_method, ot_cost=cfg.fm.ot_cost
            )
        else:
            self.fm = SchrodingerBridgeConditionalFlowMatcher(
                sigma=cfg.fm.sigma, ot_method=cfg.fm.ot_method, ot_cost=cfg.fm.ot_cost
            )
        
    def forward(self, unet_data_dict):
        # This forward pass is not directly used; training_step calls the individual models.
        return self.model(unet_data_dict)
    
    def training_step(self, batch, batch_idx):
        # Each batch is a dictionary from your dataset (from a single object)
        device = self.device  # current device

        # Move data to the proper device
        hand1_latent = batch["hand1_latent"].to(device)
        hand2_latent = batch["hand2_latent"].to(device)
        object_global_latent = batch["object_global_latent"].to(device)
        object_local_latent = batch["object_local_latent"].to(device)
        hand1_contact = batch["contact_pc_1"].to(device)
        hand2_contact = batch["contact_pc_2"].to(device)
        hand1_param = batch["hand1_param"].to(device)
        hand2_param = batch["hand2_param"].to(device)
        if self.cfg.fm.ot_cost == 'gwh':
            gwh_1 = batch["gwh_1"]
            gwh_2 = batch["gwh_2"]
        if self.cfg.fm.ot_cost == 'jac':
            jac_1 = batch["jac_1"].to(device)
            jac_2 = batch["jac_2"].to(device)

        # Build dictionary for the UViT models
        unet_data_dict = {
            "object_global_latent": object_global_latent,
            "object_local_latent": object_local_latent
        }
        # Preprocess using flow matcher:
        if self.cfg.fm.ot_cost == 'chamfer':
            # sample_map returns indices for reordering the latent features and contact points
            i, j = self.fm.sample_map(hand1_contact, hand2_contact)
        elif self.cfg.fm.ot_cost == 'euclidean':
            i, j = self.fm.sample_map(hand1_param[:, :9], hand2_param[:, :9])
        elif self.cfg.fm.ot_cost == 'gwh':
            i, j = self.fm.sample_map(gwh_1, gwh_2, device=self.device)
        elif self.cfg.fm.ot_cost == 'jac':
            i, j = self.fm.sample_map(jac_1, jac_2)
        else:
            raise ValueError(f"Unknown cost: {self.cfg.fm.ot_cost}")
        hand1_latent = hand1_latent[i]
        hand1_contact = hand1_contact[i]
        hand2_latent = hand2_latent[j]
        hand2_contact = hand2_contact[j]
        
        # Compute contact map (no gradients)
        with torch.no_grad():
            unet_data_dict["contact_map"] = sample_farthest_points(hand1_contact, K=5)[0]
        
        # Sample flow locations and noise
        t, xt, ut, eps = self.fm.sample_location_and_flow_with_plan(
            hand1_latent, hand2_latent, return_noise=True
        )
        lambda_t = self.fm.compute_lambda(t)
        unet_data_dict['hand1_latent'] = xt
        unet_data_dict['t_hand'] = t
        unet_data_dict_clone = copy.deepcopy(unet_data_dict)
        
        # Forward passes
        flow_out_dict = self.model(unet_data_dict)
        score_out_dict = self.score_model(unet_data_dict_clone)
        vt = flow_out_dict['hand_param_eps_pred']
        st = score_out_dict['hand_param_eps_pred']
        
        # Compute losses
        flow_loss = torch.mean((vt - ut) ** 2)
        score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
        loss = flow_loss + score_loss
        
        # Log training loss (both step and epoch metrics)
        log_dict = {
            'train_loss': loss,
            'flow_loss': flow_loss,
            'score_loss': score_loss
        }
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=self.pbar, sync_dist=True)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

        if self.custom_logger is not None and batch_idx == 0:
            # Log detailed information for the first batch of each epoch
            self.custom_logger(f"Epoch {self.current_epoch}, Batch {batch_idx}: loss = {loss.item():.6f}, flow_loss = {flow_loss.item():.6f}, score_loss = {score_loss.item():.6f}")

        return loss

    def configure_optimizers(self):
        # Combine parameters from both networks
        optimizer = Adam(
            list(self.model.parameters()) + list(self.score_model.parameters()),
            lr=self.hparams.learning_rate
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lr(self.hparams.warmup_total_steps)
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu, using_lbfgs):
        # Perform optimizer step as usual
        optimizer.step(closure=optimizer_closure)
        # Update EMA for both networks
        if self.global_step >= self.hparams.ema_start_step:
            # If EMA hasn't been initialized after the threshold, initialize it now.
            if not hasattr(self, "ema_initialized") or not self.ema_initialized:
                self.ema_model = copy.deepcopy(self.model)
                self.ema_score_model = copy.deepcopy(self.score_model)
                self.ema_initialized = True
                self.custom_logger("EMA started.")
            else:
                # Update EMA normally.
                ema(self.model, self.ema_model, self.hparams.ema_decay)
                ema(self.score_model, self.ema_score_model, self.hparams.ema_decay)

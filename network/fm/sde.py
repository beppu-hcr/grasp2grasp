import torch
import torch.nn as nn

class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, ode_drift, score, input_size=(3, 32, 32), sigma=1.0):
        super().__init__()
        self.drift = ode_drift
        self.score = score
        self.input_size = input_size
        self.sigma = sigma

    # Drift
    def f(self, t, y):
        y = y.view(-1, *self.input_size)
        t = t.to(y.device)
        if len(t.shape) == len(y.shape):
            t_hand = t.squeeze(1)
        else:
            t_hand = t.repeat(y.shape[0])
        assert t_hand.shape == (y.shape[0],)
        unet_data_dict = {
            "object_global_latent": y[:, :128],
            "object_local_latent": y[:, 128:128+2048*4],
            "hand1_latent": y[:, 128+2048*4:128+2048*4+768],
            "contact_map": y[:, 128+2048*4+768:].reshape(-1, 5, 3),
            "t_hand": t_hand,
        }
        drift = self.drift(unet_data_dict)['hand_param_eps_pred'] # [B, 768]
        score = self.score(unet_data_dict)['hand_param_eps_pred'] # [B, 768]
        out = torch.zeros_like(y)
        out[:, 128+2048*4:128+2048*4+768] = drift + score
        return out

    # Diffusion
    def g(self, t, y):
        out = torch.zeros_like(y)
        out[:, 128+2048*4:128+2048*4+768] += self.sigma
        return out
    
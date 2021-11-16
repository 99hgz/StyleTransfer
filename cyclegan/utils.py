class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
		

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSGanLoss(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    # NOTE c=b a=0

  def _d_loss(self, real_logit, fake_logit):
    # 1/2 * [(real-b)^2 + (fake-a)^2]
    return 0.5 * (torch.mean((real_logit - 1)**2) + torch.mean(fake_logit**2))

  def _g_loss(self, fake_logit):
    # 1/2 * (fake-c)^2
    return torch.mean((fake_logit - 1)**2)

  def forward(self, real_logit, fake_logit):
    g_loss = self._g_loss(fake_logit)
    d_loss = self._d_loss(real_logit, fake_logit)
    return d_loss, g_loss
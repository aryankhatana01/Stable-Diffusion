from collections import namedtuple
import torch
from autoencoder import AutoencoderKL
from clip_open import CLIPTextTransformer
from unet import UNetModel

class StableDiffusion:
  def __init__(self):
    self.alphas_cumprod = torch.empty(1000)
    self.model = namedtuple("DiffusionModel", ["diffusion_model"])(diffusion_model = UNetModel())
    self.first_stage_model = AutoencoderKL()
    self.cond_stage_model = namedtuple("CondStageModel", ["transformer"])(transformer = namedtuple("Transformer", ["text_model"])(text_model = CLIPTextTransformer()))

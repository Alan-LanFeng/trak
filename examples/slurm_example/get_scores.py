import torch as ch
from torchvision import models

from trak import TRAKer

model = models.resnet18(weights='DEFAULT').cuda()
model.eval()

traker = TRAKer(model=model,
                task='image_classification',
                save_dir='./slurm_example_results',
                train_set_size=10_000,  # hardcoding here
                device='cuda')

traker.finalize_scores()
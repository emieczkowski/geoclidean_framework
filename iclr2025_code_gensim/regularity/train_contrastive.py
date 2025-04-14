import os
import warnings
import hydra
from omegaconf import OmegaConf
import pyrootutils
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import DataLoader
import wandb
from utils import *
from dataset import ShapesDataset, ContrastiveTransformations, OddballDataset
warnings.filterwarnings('ignore')


def setup(cfg):
    '''Loads the relevant data.
    '''
    print("BATCH_SIZE",cfg.dataset.train.batch_size)
    use_cuda = cfg.use_gpu and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'using device: {device}')
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    # Load the behavioral data.
    behavior = OddballBehaviorModule(cfg)
    # Load the model and the dataset tforms.
    model = hydra.utils.instantiate(cfg.model, behavior=behavior)
    load_tforms = get_transforms(cfg.dataset.load)
    test_tforms = get_transforms(cfg.dataset.test)
    train_tforms = get_transforms(cfg.dataset.train)
    # Load the stimulus data for the test set (all quadrilaterals).
    imgs, stim_ids = load_quadrilaterals(behavior.stim_paths, load_tforms=load_tforms, only_canonical=False)
    test_dataset = ShapesDataset(imgs, stim_ids, behavior.symbolic_features, tforms=test_tforms)
    # Load the stimulus data for the training set (sometimes all quadrilaterals, sometimes just the canonical shapes).
    if not cfg.train_all_shapes:
        imgs, stim_ids = load_quadrilaterals(behavior.stim_paths, cfg.dataset.load, only_canonical=True)
        print(stim_ids)
    train_dataset = ShapesDataset(imgs, stim_ids, behavior.symbolic_features, tforms=train_tforms)
    # Load the datasets.
    test_loader = DataLoader(test_dataset, batch_size=18,shuffle=cfg.dataset.test.shuffle, num_workers=3, pin_memory=True)
    train_loader = DataLoader(train_dataset, batch_size=18, shuffle=cfg.dataset.train.shuffle, num_workers=3, pin_memory=True)
    return model, train_loader, test_loader, device

# project root setup
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
@hydra.main(version_base=None, config_path='config', config_name='train_encoder')
def run(cfg):
    model, train_loader, test_loader, device = setup(cfg)
    callbacks = instantiate_modules(cfg.get('callbacks'))
    # Instantiate the logger with the appropriate names.
    hid_dim = cfg.model.hidden_dim
    train_all = cfg.train_all_shapes
    logger = instantiate_modules(cfg.get('logger'))
    logger[0].experiment.config.update({'hidden_dim': hid_dim, 
                                        'train_all_shapes': train_all}, 
                                        allow_val_change=True)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    print(type(trainer))
    model = trainer.fit(model, train_loader, test_loader)
    logger[0].experiment.finish()

if __name__=='__main__':
    run()
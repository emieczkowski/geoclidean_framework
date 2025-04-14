from einops import repeat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import optim
from scipy import stats
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import cornet



class Encoder_cornet(nn.Module):
	def __init__(self,hidden_dim=128):
		super(Encoder_cornet, self).__init__()
		cornet_model=getattr(cornet, 'cornet_s')
		self.cornet= cornet_model(pretrained=True, map_location=None)
		self.fc=nn.Linear(1000,hidden_dim)
	def forward(self,x):
		cornet_output=self.cornet(x)
		return self.fc(cornet_output)
	
	

class ContrastiveLitModule(pl.LightningModule):
	def __init__(self, behavior, hidden_dim=128, lr=5e-4, temperature=1e-2, weight_decay=1e-6, max_epochs=13, symbolic_target=True):
		super().__init__()
		self.save_hyperparameters()
		self.behavior = behavior
		assert self.hparams.temperature > 0.0, 'Temperature must be a positive float!'
		self.net = Encoder_cornet(hidden_dim)
		#self.net= SimCLR(hidden_dim=hidden_dim )
		#self.net=ConvEncoder()
		self.outputs = [] # Track reps from validation step.
		self.shapes = [] # Track the sfhapes for each trial.
		self.pred_feat = [] # Track MDL for dreamcoder trials and shape id for DMTS trials.

	def configure_optimizers(self):
		optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
		lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr/self.hparams.max_epochs)
		return [optimizer], [lr_scheduler]

	def loss(self, batch, mode='train', batch_idx=0): 
		
		return self.symbolic_loss(batch, mode=mode, batch_idx=batch_idx)
		#return self.info_nce_loss(batch,mode=mode,batch_idx=batch_idx)
	
		

	def info_nce_loss(self, batch, mode='train', batch_idx=0):
		imgs, stim_ids, _ = batch
		if mode=='train':
			imgs, stim_ids = torch.cat(imgs, dim=0).float(), torch.cat(stim_ids, dim=0).float()
		if imgs.shape[0]!=stim_ids.shape[0]:
			raise Exception('Images and stimulus ids must have the same length!')
		reps = self.net(imgs)
		# Calculate cosine similarity
		cos_sim = F.cosine_similarity(reps[:,None,:], reps[None,:,:], dim=-1)
		# mask out the item matches
		stim_ids = stim_ids.T.ravel()
		match_mask = (stim_ids[:,None]==stim_ids[None,:]).to(cos_sim.device)
		identity_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
		pos_mask = identity_mask.roll(shifts=cos_sim.shape[0]//2, dims=0)
		self_mask = (match_mask | identity_mask) & (~pos_mask)
		# Mask out cosine similarity to itself.
		cos_sim.masked_fill_(self_mask, -9e15)
		# Find positive example -> batch_size//2 away from the original example
		cos_sim = cos_sim / self.hparams.temperature
		nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
		nll = nll.mean()
		# Logging loss
		self.log(mode+'_loss', nll)
		# Get ranking position of positive example
		comb_sim = torch.cat([cos_sim[pos_mask][:,None], cos_sim.masked_fill(pos_mask, -9e15)], dim=-1)
		sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
		# Logging ranking metrics
		self.log(mode+'_acc_top1', (sim_argsort == 0).float().mean())
		self.log(mode+'_acc_top5', (sim_argsort < 5).float().mean())
		self.log(mode+'_acc_mean_pos', 1+sim_argsort.float().mean())
		return nll
	
	def symbolic_loss(self, batch, mode='train', batch_idx=0):
		imgs, pred, feature_reps = batch # pred for DreamCoder is the MDL, and for DMTS is the shape id.
		if mode=='train':
			imgs, pred, feature_reps = torch.cat(imgs, dim=0), torch.cat(pred, dim=0).float(), torch.cat(feature_reps, dim=0).float()
		reps = self.net(imgs)
		# Compute the pairwise similarity between the 
		symbol_sim = torch.cdist(feature_reps, feature_reps) #F.cosine_similarity(feature_reps[:,None,:], feature_reps[None,:,:], dim=-1) #
		rep_sim = torch.cdist(reps, reps) #F.cosine_similarity(reps[:,None,:], reps[None,:,:], dim=-1) #
		# Compute the loss between the two similarity matrices.
		nll = F.mse_loss(rep_sim, symbol_sim)
		# Logging loss
		self.log(mode+'_loss', nll)
		return nll

	def training_step(self, batch, batch_idx):
		return self.loss(batch, mode='train')

	def validation_step(self, batch, batch_idx):
		imgs, _, _ = batch
		#imgs = torch.cat(imgs, dim=0)
		reps = self.net(imgs)
		self.outputs.append(reps)
		#return self.loss(batch, mode='val')
	
	def on_validation_epoch_end(self):
		print("DIRECTORY")
		reps = torch.cat(self.outputs).detach().cpu().numpy()
		trial_reps = reps[self.behavior.all_trials.ravel()].reshape(-1,6,reps.shape[-1])
		dists = np.linalg.norm(trial_reps-trial_reps.mean(axis=1, keepdims=True), axis=2)
		outliers = dists.argmax(axis=1)
		model_success = self.behavior.all_behavior['outlierPosition'].values==outliers
		trial_shapes = self.behavior.all_behavior['shape'].values
		error_rates, human_r_value, human_p_value, human_rmse, monkey_r_value, monkey_p_value, monkey_rmse = self.behavior.correlate_behavior(model_success, trial_shapes)
		self.log('human_r_value', human_r_value)
		self.log('human_p_value', human_p_value)
		self.log('monkey_r_value', monkey_r_value)
		self.log('monkey_p_value', monkey_p_value)
		self.log('human_rmse', human_rmse)
		self.log('monkey_rmse', monkey_rmse)
		ranks = torch.arange(error_rates.shape[0])
		df = pd.DataFrame({'Irregularity': ranks, 'Error Rate': error_rates})
		data_path=self.logger.save_dir+'/'+'{}_data_dict.npz'.format(self.current_epoch)
		np.savez(data_path,error_rates=error_rates,human_r_value=human_r_value,human_p_value=human_p_value,monkey_r_value=monkey_r_value,monkey_p_value=monkey_p_value,irregularity=ranks,model_success=model_success,trial_shapes=trial_shapes)
		fig = px.scatter(df, x='Irregularity', y='Error Rate', trendline='ols', trendline_color_override='red', title=f'Regularity vs. Error Rate')
		wandb.log({'error rates': fig})
		print(error_rates)
		print(human_r_value,human_p_value)
		# Log the p-value of the slope of a linear model fit being greater than 0.
		_, _, _, p_value, _ = stats.linregress(ranks, error_rates, alternative='greater')
		self.log('slope_p_value', p_value)
		self.outputs = []
		if self.current_epoch>=20:
			raise KeyboardInterrupt
		
import glob
import os
from typing import List
import itertools
import pickle

from PIL import Image
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cosine, cdist, pdist
import hydra
import torchvision.transforms.functional as TF
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms

from dataset import ContrastiveTransformations

# A class to manage loading and retrieving the behavioral data for the oddball task.
class OddballBehaviorModule:
	def __init__(self, cfg):
		self.cfg = cfg
		self.shapes = self.cfg.shapes
		self.humans = self.cfg.human_conditions
		self.monkeys = self.cfg.monkey_conditions
		human_trials, monkey_trials, human_behavior, monkey_behavior, stim_paths, stim_df, features = self.load_data()
		self.stim_paths = stim_paths
		self.stim_df = stim_df
		self.symbolic_features = features
		# Compute the behavioral error rates for each condition (human, monkey, combined).
		self.human_error_rates = self.compute_error_rates(human_behavior['success'].values, human_behavior['shape'].values)
		self.monkey_error_rates = self.compute_error_rates(monkey_behavior['success'].values, monkey_behavior['shape'].values)
		self.all_trials = np.concatenate([human_trials, monkey_trials])
		self.all_behavior = pd.concat([human_behavior, monkey_behavior])
		self.combined_error_rates = self.compute_error_rates(self.all_behavior['success'].values, self.all_behavior['shape'].values)
		# Generate a set of synthetic oddball trials.
		if self.cfg.gen_synthetic_trials:
			synth_trials, synth_ys, synth_shapes = self.gen_oddball_trials(self.stim_df)
			good_inds = self.get_good_trial_inds(synth_trials) # Filter out trials that match the behavioral data.
			self.synth_trials, self.synth_ys, self.synth_shapes = synth_trials[good_inds], synth_ys[good_inds], synth_shapes[good_inds]

		# Log behavior performance during training.
		self.model_error_rates = []
		self.human_r_values = []
		self.human_p_values = []
		self.monkey_r_values = []
		self.monkey_p_values = []
		self.human_rmse = []
		self.monkey_rmse = []
	
	def load_data(self):
		'''Load all behavioral data for humans and monkeys.
		'''
		human_trials, human_behavior, stim_paths, stim_df, features = self.load_all_conditions(self.humans, self.cfg.paths.data_dir)
		monkey_trials, monkey_behavior, _, _, _ = self.load_all_conditions(self.monkeys, self.cfg.paths.data_dir)
		#### NOTE: CHANGES TRIAL SAMPLING TO ONLY INCLUDE IRREGULAR TRIALS. ####
		#human_trials = human_trials[human_behavior.majorityOfMostRegularShapes.values==False]
		#human_behavior = human_behavior[human_behavior.majorityOfMostRegularShapes.values==False]
		#monkey_trials = monkey_trials[monkey_behavior.majorityOfMostRegularShapes.values==False]
		#monkey_behavior = monkey_behavior[monkey_behavior.majorityOfMostRegularShapes.values==False]
		#### NOTE END ####

		#### NOTE: BALANCE IRREGULAR AND REGULAR TRIALS. ####
		# humans first.
		#maj_irreg_inds = np.random.permutation(np.argwhere(human_behavior.majorityOfMostRegularShapes.values==False)).ravel()
		#maj_reg_inds = np.random.permutation(np.argwhere(human_behavior.majorityOfMostRegularShapes.values==True)).ravel()
		#n_samples = min(len(maj_irreg_inds), len(maj_reg_inds))
		#human_trials = np.vstack([human_trials[maj_irreg_inds[:n_samples]], human_trials[maj_reg_inds[:n_samples]]])
		#human_behavior = pd.concat([human_behavior.iloc[maj_irreg_inds[:n_samples]], human_behavior.iloc[maj_reg_inds[:n_samples]]])
		## monkeys next.
		#maj_irreg_inds = np.random.permutation(np.argwhere(monkey_behavior.majorityOfMostRegularShapes.values==False)).ravel()
		#maj_reg_inds = np.random.permutation(np.argwhere(monkey_behavior.majorityOfMostRegularShapes.values==True)).ravel()
		#n_samples = min(len(maj_irreg_inds), len(maj_reg_inds))
		#monkey_trials = np.vstack([monkey_trials[maj_irreg_inds[:n_samples]], monkey_trials[maj_reg_inds[:n_samples]]])
		#monkey_behavior = pd.concat([monkey_behavior.iloc[maj_irreg_inds[:n_samples]], monkey_behavior.iloc[maj_reg_inds[:n_samples]]])
		#### NOTE END ####
		return human_trials, monkey_trials, human_behavior, monkey_behavior, stim_paths, stim_df, features

	def compute_error_rates(self, success, trial_shapes):
		''' Compute the error rates for each shape.
		trial_shapes: array of shape codes for each trial.
		success: array of success/failure for each trial.
		'''
		accs = np.zeros(len(self.shapes))
		for i, shape in enumerate(self.shapes):
			accs[i] = success[trial_shapes==shape].astype(int).mean()
		return 1-accs

	def correlate_behavior(self, model_success, trial_shapes, cond='logits'):
		''' Correlate the model representations with the behavioral data.
		model_reps: array of model representations for each stimulus.
		'''
		error_rates = self.compute_error_rates(model_success, trial_shapes) #self.compute_error_rates(self.all_behavior['shape'].values, model_success)
		# Compute some correlation values.
		_, _, human_r_value, human_p_value, _ = stats.linregress(self.human_error_rates, error_rates, alternative='greater')
		_, _, monkey_r_value, monkey_p_value, _ = stats.linregress(self.monkey_error_rates, error_rates, alternative='greater')
		# Compute the RMSE.
		xs = np.arange(len(error_rates))
		slope, intercept, _, _, _ = stats.linregress(xs, error_rates)
		# Predict the monkey and human error rates and compute the RMSE of the model fit.
		preds = slope*xs + intercept
		monkey_rmse = np.sqrt(np.mean((preds-self.monkey_error_rates)**2))
		human_rmse = np.sqrt(np.mean((preds-self.human_error_rates)**2))
		# Store the relevant results.
		if cond=='logits':
			self.model_error_rates.append(error_rates)
			self.human_r_values.append(human_r_value)
			self.human_p_values.append(human_p_value)
			self.monkey_r_values.append(monkey_r_value)
			self.monkey_p_values.append(monkey_p_value)
			self.human_rmse.append(human_rmse)
			self.monkey_rmse.append(monkey_rmse)
		return error_rates, human_r_value, human_p_value, human_rmse, monkey_r_value, monkey_p_value, monkey_rmse
	
	def load_all_conditions(self, conditions, data_dir):
		'''Load all of the experimental conditions.
		conditions: list of condition names.
		data_dir: path to the data directory.
		'''
		# Get the stimulus paths.
		stim_paths = sorted(glob.glob(os.path.join(data_dir, 'oddball', 'stimuli', '*.png')))
		stim_df, features = self.get_stimuli_metadata(stim_paths)
		# Aggregate the trial data across all conditions.
		all_trials = []
		all_behavior = []
		for cond in conditions:
			trials, behavior = self.get_condition_trials(cond, stim_df, data_dir)
			all_trials.append(trials)
			all_behavior.append(behavior)
		all_trials = np.vstack(all_trials)
		all_behavior = pd.concat(all_behavior)
		return all_trials, all_behavior, stim_paths, stim_df, features

	def get_stimuli_metadata(self, stim_paths):
		'''Get metadata about all the stimuli from the file paths.
		stim_paths: list of paths to the stimulus images.
		'''
		stim_info = [path.split('/')[-1].split('.')[0].split('_') for path in stim_paths]
		df = pd.DataFrame(np.array(stim_info)[:, 1:], columns=['stimShape', 'isOutlier', 'stimSize', 'stimRotation'])
		df['outlierType'] = df['isOutlier'].str.replace('\D+', '')
		df['isOutlier'] = df['isOutlier'].str.replace('\d+', '')
		df['isOutlier'] = df['isOutlier']=='outlier'
		# Load the stimulus symbolic features and sort them so that they align with the stimulus df.
		shape_codes = np.load(os.path.join(self.cfg.paths.data_dir, 'oddball', 'codes.npy')).astype('S13')
		features = np.load(os.path.join(self.cfg.paths.data_dir, 'oddball', 'reps.npy'))
		is_outlier = np.tile(['False', 'True', 'True', 'True', 'True'], (11,)).astype('S5')
		outlier_type = np.tile(['', '0', '1', '2', '3'], (11,)).astype('S1')
		temp = np.core.defchararray.add(shape_codes, outlier_type)
		feat_mdata = np.core.defchararray.add(temp, is_outlier)
		stim_mdata = (df.stimShape + df.outlierType.astype(str) + df.isOutlier.astype(str)).astype('S19')
		feat_mdata = feat_mdata.repeat(36)
		sort_inds = np.argsort(feat_mdata)
		feat_mdata = feat_mdata[sort_inds]
		features = features.repeat(36, axis=0)[sort_inds]
		return df, features

	def get_condition_trials(self, condition, df, data_dir):
		'''Get all behavioral trials for a provided data condition.
		condition: name of the condition.
		df: dataframe containing the stimuli metadata.
		data_dir: path to the data directory.
		'''
		# Load the behavioral data.
		behavior = pd.read_csv(os.path.join(data_dir, 'oddball', 'behavior', condition+'.csv'))
		rot_map = {-25:1, -15:2, -5:3, 5:4, 15:5, 25:6,
				1:1, 2:2, 3:3, 4:4, 5:5, 6:6} # for some baboons.
		dil_map = {.875:1, .925:2, .975:3, 1.025:4, 1.075:5, 1.125:6, # all other conditions.
				0.75:1, 0.85:2, 0.95:3, 1.05:4, 1.15:5, 1.25:6,    # for french1 condition.
				1:1, 2:2, 3:3, 4:4, 5:5, 6:6}                      # for some baboons.  
		rot_cols = ['rot'+str(i) for i in range(1,7)]
		dil_cols = ['dil'+str(i) for i in range(1,7)]
		behavior[rot_cols] = behavior[rot_cols].apply(lambda x: [rot_map[i] for i in x])
		behavior[dil_cols] = behavior[dil_cols].apply(lambda x: [dil_map[i] for i in x])
		df['stim_id'] = df.stimShape + df.stimSize.astype(str) + df.stimRotation.astype(str)
		# Construct an array that uniquely identifies each stimulus.
		rot = behavior[['rot'+str(i) for i in range(1,7)]].values.astype(str).ravel()
		dil = behavior[['dil'+str(i) for i in range(1,7)]].values.astype(str).ravel()
		shape = behavior['shape'].values.repeat(6).astype(str)
		outlier_type = behavior['outlierType'].values.repeat(6)
		is_regular = behavior['majorityOfMostRegularShapes'].values.repeat(6)
		stim_ids = np.char.add(np.char.add(shape, dil), rot)
		# Get the file indices of both perturbed and regular stimuli across all trials
		all_trials = np.stack([np.argwhere(df.stim_id.values==id).ravel() for id in stim_ids])
		perturbed_stimuli = all_trials[np.arange(len(all_trials)), outlier_type]
		regular_stimuli = all_trials[np.arange(len(all_trials)), np.full(len(all_trials), -1)]
		all_trials = np.stack([perturbed_stimuli, regular_stimuli]).T
		# Get the correct indices for each stimulus (either regular or perturbed 1/2/3 depending on trial/outlier-type).
		oddball_inds = np.zeros([behavior.shape[0], 6])
		position = behavior['outlierPosition'].values
		oddball_inds[np.arange(len(oddball_inds)), position] = 1
		oddball_inds = np.logical_xor(oddball_inds, is_regular.reshape(-1,6)) # invert oddball inds if the outlier is the regular shape.
		all_trials = all_trials[np.arange(len(all_trials)), oddball_inds.ravel().astype(int)].reshape(-1,6)
		# Correct the choice values for babboons whose indexing starts at 1 for some reason.
		if behavior.choice.values.max()==6:
			behavior.choice = behavior.choice-1
		return all_trials, behavior

	def gen_oddball_trials(self, df, N=1000):
		'''Generate a set of synthetic oddball trials.
		df: dataframe containing the stimuli metadata.
		N: number of trials to generate.
		'''
		all_trials = []
		trial_shapes = []
		for shape in df.stimShape.unique():
			reg_inds = df[(df.stimShape==shape) & (df.isOutlier==False)].index.values  # inds of regular stimuli.
			for outlier_type in df.outlierType.unique():
				# Skip the outlier context.
				if len(outlier_type)==0:
					continue
				# Generate the set of possible choice arrays for this stimulus condition.
				inds = (df.stimShape==shape) & (df.outlierType==outlier_type) & (df.isOutlier==True)
				perturb_inds = df[inds].index.values # inds of perturbed stimuli.
				all_trials.append(self.get_combinations(reg_inds, perturb_inds, N=N)) # regular trials
				all_trials.append(self.get_combinations(perturb_inds, reg_inds, N=N)) # perturbed trials
				trial_shapes.append(np.repeat(shape, N*2)) # shape metadata
				#trial_shapes.append(np.repeat(shape, N)) # shape metadata
		all_trials = np.vstack(all_trials)
		trial_shapes = np.concatenate(trial_shapes)
		trial_answers = all_trials[:,0]
		shuffled_trials = np.apply_along_axis(np.random.permutation, axis=1, arr=all_trials)
		answers = (trial_answers.repeat(6).reshape(-1,6)==shuffled_trials).astype(int)
		return shuffled_trials, answers, trial_shapes
	
	def get_combinations(self, option_inds, oddball_inds, N=1000):
		'''Get all possible combinations of regular and perturbed stimuli for a given stimulus condition.
		options_inds: indices of regular stimuli for current condition.
		oddball_inds: indices of oddball stimuli for current condition.
		'''
		# Generate the set of possible choice arrays for this stimulus condition.
		options = np.array(list(itertools.combinations(option_inds, 5)))
		# Sample a subset of items.
		inds = np.random.permutation(len(options))[:N]
		options = options[inds]
		# Now generate all permutations of the selected options arrays.
		oddballs = np.repeat(oddball_inds, len(options))
		options_final = np.tile(options, (len(oddball_inds),1))
		all_trails = np.hstack([oddballs.reshape(-1,1), options_final])
		trials = all_trails[np.random.choice(len(all_trails), N, replace=False)] # sample N random trials 
		return trials
	
	def get_good_trial_inds(self, synthetic_trials):
		'''Get the indices of the synthetic trials that weren't used in the behavioral data.
		synthetic_trials: array of candidate synthetic trials.
		'''
		def get_trial_codes(trials):
			trials_sorted = np.sort(trials, axis=1).astype('str')
			trials_padded = np.char.zfill(trials_sorted, 4)
			trial_codes = np.apply_along_axis(lambda x: ' '.join(x), axis=1, arr=trials_padded)
			return trial_codes
		# Filter out trials that match any of the trials from the behavioral data.
		synth_trial_codes = get_trial_codes(synthetic_trials)
		all_trial_codes = get_trial_codes(self.all_trials)
		bad_inds = np.isin(synth_trial_codes, all_trial_codes)
		return ~bad_inds

def get_imgs(stim_paths, load_tforms=None):
	stim_tensors = torch.stack([TF.pil_to_tensor(Image.open(path).convert('RGB')).float()/255 for path in stim_paths])
	if load_tforms is not None:
		stim_tensors = load_tforms(stim_tensors)
	stim_tensors[stim_tensors>0.1] = 1
	stim_tensors[stim_tensors<=0.1] = 0
	return stim_tensors

def load_quadrilaterals(stim_paths, load_tforms, only_canonical=False, n_repeats=10):
	# Convert file paths to shape codes.
	stim_codes = [' '.join(path.split('_')[1:3]) for path in stim_paths]
	# If only training on canonical shapes, filter the stimulus codes and paths.
	if only_canonical:
		stim_codes = [code for code in stim_codes if 'reference' in code]*n_repeats
		stim_paths = [path for path in stim_paths if 'reference' in path]*n_repeats
	# Convert stim_codes to unique integer labels.
	_, stim_ids = np.unique(stim_codes, return_inverse=True)
	imgs = get_imgs(stim_paths, load_tforms=load_tforms)
	stim_ids = torch.tensor(stim_ids).reshape(-1,1)
	return imgs, stim_ids

def instantiate_modules(module_cfgs):
	modules = []
	for _, module in module_cfgs.items():
		if '_target_' in module:
			modules.append(hydra.utils.instantiate(module))
	return modules

def get_transforms(tform_cfg):
	# If no transforms were specified, return None.
	if not tform_cfg.tforms:
		return None
	# Otherwise, load them sequentially.
	tforms = []
	for _, conf in tform_cfg.tforms.items():
		if '_target_' in conf:
			tforms.append(hydra.utils.instantiate(conf))
	tforms = transforms.Compose(tforms)
	# And wrap them in a contrastive transformation if necessary.
	if tform_cfg.contrastive_tform:
		return ContrastiveTransformations(tforms, n_views=2)
	return tforms
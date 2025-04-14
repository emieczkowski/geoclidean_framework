import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset


class ContrastiveTransformations(nn.Module):
	def __init__(self, base_transforms, n_views=2):
		self.base_transforms = base_transforms
		self.n_views = n_views

	def __call__(self, x):
		xs = [self.base_transforms(x) for i in range(self.n_views)]
		#ids = torch.cat([ids for _ in range(self.n_views)])
		return xs #, ids
	
class MakeEdges(object):
	def __init__(self, crop_min=212, crop_max=236, target_size=225):
		self.crop_min = crop_min
		self.crop_max = crop_max
		self.target_size = target_size
	def __call__(self, tensor):
		resize = transforms.Resize([self.target_size,self.target_size], antialias=True)
		min_crop = resize(transforms.CenterCrop([self.crop_min,self.crop_min])(tensor))
		max_crop = resize(transforms.CenterCrop([self.crop_max,self.crop_max])(tensor))
		return min_crop-max_crop
	def __repr__(self):
		return self.__class__.__name__ + f'Crop min={self.crop_max}, Crop max={self.crop_max}'

class RandomCenterCrop(nn.Module):
	def __init__(self, crop_min=150, crop_max=256):
		self.crop_min = crop_min
		self.crop_max = crop_max
	def __call__(self, tensor):
		crop = torch.randint(self.crop_min, self.crop_max, (1,)).item()
		return transforms.CenterCrop([crop,crop])(tensor)
	def __repr__(self):
		return self.__class__.__name__ + f'Crop min={self.crop_max}, Crop max={self.crop_max}'
	
class ShapesDataset(Dataset):
	def __init__(self, stimuli, stim_ids, features, tforms=None):
		self.xs = stimuli
		self.ids = stim_ids
		self.features = features.astype('float32')
		self.tforms = tforms
	
	def __len__(self):
		return len(self.xs)

	# TODO: fix this so that it works with the contrastive transformations.
	# IDEA: only pass xs to contrastive transforms and artificially tile the batch_ids and labels.
	def __getitem__(self, idx):
		xs, batch_ids, features = self.xs[idx], self.ids[idx], torch.tensor(self.features[idx])
		if self.tforms:
			if isinstance(self.tforms, ContrastiveTransformations):
				xs = self.tforms(xs)
				batch_ids = [batch_ids for _ in range(self.tforms.n_views)]
				features = [features for _ in range(self.tforms.n_views)]
			else:
				xs = self.tforms(xs)
		return xs, batch_ids, features
	
class OddballDataset(Dataset):
	def __init__(self, stimuli, ys, all_trials, shapes, tforms=None):
		self.xs = stimuli
		self.ys = ys
		self.all_trials = all_trials
		self.shapes = shapes
		self.tforms = tforms
	
	def __len__(self):
		return len(self.ys)

	def __getitem__(self, idx):
		y, trial_ids, shape = self.ys[idx], self.all_trials[idx], self.shapes[idx]
		xs = self.xs[trial_ids]
		if self.tforms:
			xs = self.tforms(xs)
		return xs, y, shape
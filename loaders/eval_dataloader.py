import random
import torch
import torch.multiprocessing as mp

PROCESSORS = 6

class EvalDataloader:

	def __init__(self, dataset, samples):

		self.dataset = dataset
		self.num_samples = samples

		self.index = 0

		pool = mp.Pool(PROCESSORS)
		chosen = random.sample(range(len(self.dataset)), self.num_samples)

		self.processed = pool.map(self.dataset.__getitem__, chosen)
		self.processed = [(img.unsqueeze(0), lbl.unsqueeze(0)) for (img, lbl) in self.processed]

		self.unprocessed = pool.map(self.dataset.get_original, chosen)

		self.data = list(zip(self.processed, self.unprocessed))

	def next(self):
		d = self.data[self.index]
		self.index += 1
		return d

	def reset(self):
		self.__init__(self.dataset, self.num_samples)

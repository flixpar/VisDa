class Namespace:
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)
	def print(self):
		for key, val in self.__dict__.items():
			print("{}:\t{}".format(key, val))

def poly_lr_scheduler(optimizer, init_lr, it, lr_decay_iter=1, max_iter=100, power=0.9):
	if it % lr_decay_iter or it > max_iter:
		return optimizer

	lr = init_lr*(1 - it/max_iter)**power
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return lr

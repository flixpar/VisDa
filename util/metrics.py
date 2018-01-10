import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import metrics as skmetrics

def _fast_hist(label_true, label_pred, n_class):
	mask = (label_true >= 0) & (label_true < n_class)
	hist = np.bincount(
		n_class * label_true[mask].astype(int) +
		label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
	return hist


def scores(label_trues, label_preds, n_class):
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
	acc = np.diag(hist).sum() / hist.sum()
	acc_cls = np.diag(hist) / hist.sum(axis=1)
	acc_cls = np.nanmean(acc_cls)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	freq = hist.sum(axis=1) / hist.sum()
	fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
	cls_iu = dict(zip(range(n_class), iu))

	return {'Overall Acc': acc,
			'Mean Acc': acc_cls,
			'FreqW Acc': fwavacc,
			'Mean IoU': mean_iu,}, cls_iu

def miou(label_trues, label_preds, n_class):
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	mean_iu = np.nanmean(iu)
	return mean_iu

def class_iou(label_trues, label_preds, n_class):
	hist = np.zeros((n_class, n_class))
	for lt, lp in zip(label_trues, label_preds):
		hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
	iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
	cls_iu = iu
	# cls_iu = dict(zip(range(n_class), iu))
	return cls_iu

def print_scores(pred, gt, n_classes, i=0):
	if i: print("### Image {} ###".format(i))
	score, class_iou = scores(gt, pred, n_classes)
	for key, val in score.items():
		print("{}:\t{}".format(key, val))
	for key, val in class_iou.items():
		if not np.isnan(val):
			print("{}:\t{}".format(key, val))
	print()

def plotConfusionMatrix(gt, pred, num_classes, fn="confusion_matrix.png"):
	cfm = skmetrics.confusion_matrix(gt.flatten(), pred.flatten())

	plt.figure()

	# normalize and display
	cfm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
	plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
	
	# setup the title and axes
	plt.title("Normalized Confusion Matrix")
	plt.colorbar()
	tick_marks = np.arange(num_classes)
	plt.xticks(tick_marks, range(num_classes), rotation=45)
	plt.yticks(tick_marks, range(num_classes))

	# add values
	thresh = cfm.max() / 2.
	for i, j in itertools.product(range(cfm.shape[0]), range(cfm.shape[1])):
		plt.text(j, i, cfm[i, j], size="x-small", horizontalalignment="center", color = ("white" if cfm[i, j] > thresh else "black"))

	# label the axes
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	
	# save image and display
	plt.savefig(fn, dpi=100)
	plt.show()

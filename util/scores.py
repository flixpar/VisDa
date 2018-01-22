import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from util.metrics import calc_miou, calc_class_iou, confusion_matrix


class Scorer:

	def __init__(self, selected_metrics, num_classes):

		self.selected_metrics = selected_metrics
		self.num_classes = num_classes

		self.use_miou = "miou" in self.selected_metrics
		self.use_clsiou = "cls_iou" in self.selected_metrics
		self.use_cfm = "cfm" in self.selected_metrics
		self.use_matches = "classmatch" in self.selected_metrics
		
		self.miou = []
		self.cls_iou = []
		self.cfm = []
		self.class_matches = []

		self.size = 0

	def update(self, gt, pred):
		
		iter_cls_iou = calc_class_iou(gt, pred, self.num_classes)
		
		if self.use_miou:
			iter_miou = np.nanmean(iter_cls_iou)
			self.miou.append(iter_miou)

		if self.use_clsiou:
			# iter_cls_iou = np.nan_to_num(iter_cls_iou)
			self.cls_iou.append(iter_cls_iou)

		if self.use_cfm:
			iter_cfm = confusion_matrix(gt.flatten(), pred.flatten(), self.num_classes, normalize=False)
			self.cfm.append(iter_cfm)

		if self.use_matches:
			k = 0.01 * pred.size

			p1,p2 = np.unique(pred, return_counts=True)
			p = dict(zip(p1,p2))

			g1,g2 = np.unique(gt.astype(np.uint8), return_counts=True)
			g = dict(zip(g1,g2))

			matches = [(i in g1) == (i in p1) for i in range(self.num_classes)]
			for i in range(self.num_classes):
				if not i in g.keys(): g[i] = 0
				if not i in p.keys(): p[i] = 0
			matches = [matches[i] if ((g[i]>k) == (p[i]>k)) else False for i in range(len(matches))]

			self.class_matches.append(matches)

		self.size += 1

	def to_string(self):
		
		a, b, c, d = self.final_scores()
		out = "Scores averaged over {} images:\n".format(self.size)

		if self.use_miou:
			out += "mIOU: {}\n".format(a)

		if self.use_clsiou:
			out += "class IOU: {}\n".format(b)

		if self.use_matches:
			out += "matches: {}\n".format(d)

		return out

	def latest_to_string(self):

		out = ""

		if self.use_miou:
			out += "mIOU: {}\n".format(self.miou[-1])

		if self.use_clsiou:
			out += "class IOU: {}\n".format(list(self.cls_iou[-1]))

		if self.use_matches:
			out += "matches: {}\n".format((list(self.class_matches[-1])))

		return out

	def final_scores(self):
		res = []

		if self.use_miou:
			meaniou = np.asarray(self.miou).mean()
			stdeviou = np.asarray(self.miou).std()
			res.append((meaniou, stdeviou))

		if self.use_clsiou:
			total_clsiou = np.nan_to_num(self.cls_iou.copy())
			total_clsiou = np.mean(total_clsiou, axis=0)
			res.append(total_clsiou)

		if self.use_cfm:
			out_cfm = np.sum(self.cfm, axis=0).astype(np.float64)
			out_cfm = out_cfm / out_cfm.sum(axis=1)[:, np.newaxis]
			res.append(out_cfm)

		if self.use_matches:
			res.append(self.class_matches)

		return tuple(res)


from pathlib import Path
import csv

experiment_names = ["6"]

def main():

	for name in experiment_names:

		folder = Path("saves_" + name)
		#log = open(folder/"train.log", 'r')
		log = open("train.log", 'r')

		log_lines = [line.rstrip('\n') for line in log if "miou" in line]
		scores = {parse_score_line(line) for line in log_lines}
		print(scores)
		write_scores(scores, name)

def parse_score_line(line):
	parts = line.split(",")
	it = int(parts[0].split(' ')[-1])
	iou = float(parts[1][9:])
	return it, iou

def write_scores(scores, name):
	with open("{}.csv".format(name), 'w', newline='') as csvfile:
		csv_writer = csv.writer(csvfile, delimiter=',')
		for score in sorted(scores):
			print(score)
			csv_writer.writerow(list(score))

if __name__ == "__main__":
	main()
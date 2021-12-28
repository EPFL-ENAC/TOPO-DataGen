import os
import pdb
import wget
import argparse
import zipfile
import pandas as pd
from glob import glob


def main():
	args = argparse.ArgumentParser()
	args.add_argument('csv_dir', type=str, help='Directory to csv files.')
	args = args.parse_args()

	csv_ls = sorted(glob(os.path.abspath(os.path.join(args.csv_dir, '*.csv'))))
	assert len(csv_ls)
	print("{:d} csv files are found at {:s}.".format(len(csv_ls), os.path.abspath(args.csv_dir)))
	[print(csv_file) for csv_file in csv_ls]
	print('\n')

	for csv_path in csv_ls:
		df = pd.read_csv(csv_path, header=None)
		link_ls = df.to_numpy()

		for link in link_ls:
			link = str(link[0])
			save_dirname = '-'.join(os.path.basename(csv_path).split('.')[0].split('-')[0:1] +
									os.path.basename(csv_path).split('.')[0].split('-')[3:])
			save_dir = os.path.abspath(os.path.join(args.csv_dir, save_dirname))
			os.makedirs(save_dir, exist_ok=True)

			save_path = os.path.join(save_dir, os.path.basename(link))
			print('Saving {:s} to {:s}'.format(link, save_path))
			wget.download(link, save_path)
			print('\n')

			if link.endswith('.zip'):
				with zipfile.ZipFile(save_path, 'r') as zip_ref:
					zip_ref.extractall(save_dir)
				os.remove(save_path)


if __name__ == '__main__':
	main()

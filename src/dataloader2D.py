import cv2
import torch
import os

import pandas as pd
import numpy as np

from tqdm import tqdm 

class OSICDataset(Dataset):
	def __init__(self, csv_file, root_dir, transform=None):
		self.csv = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self):
		pass

	def __getitem__(self, idx):
		patient = self.csv.loc[idx, :]

		img_path = os.path.join()
		img = cv2.imread()
		
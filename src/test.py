import numpy as np
import pandas as pd

import os
import cv2

if __name__ == "__main__":
	train = pd.read_csv("../input/interim/train_fe.csv")
	print(train.head())
	patient = train.loc[1, :]
	print(patient)

	path = "../input/raw/images/train"
	img_path = os.path.join(path, patient.Patient)
	img_path = img_path + "/1.jpg"
	img = cv2.imread(img_path)
	print(img.shape)
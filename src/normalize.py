import pandas as pd

class Normalize:
	def __init__(self, df, cols):
		self.df = df
		self.cols = cols

	def scale_features(self):
		for col in cols:
			df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

if __name__ == "__main__":
	df_train = pd.read_csv("../input/interim/train_fe_ohe.csv")
	df_test = pd.read_csv("../input/interim/test_fe_ohe.csv")
	df_ss = pd.read_csv("../input/interim/ss_fe_ohe.csv")

	
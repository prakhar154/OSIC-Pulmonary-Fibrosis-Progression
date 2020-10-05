import pandas as pd

class Normalize:
	def __init__(self, df, cols):
		self.df = df
		self.cols = cols

	def scale_features(self):
		df_ = self.df.copy(deep=True)
		for col in self.cols:
			df_[col] = (df_[col] - df_[col].min()) / (df_[col].max() - df_[col].min())
		return df_

if __name__ == "__main__":
	df_train = pd.read_csv("../input/interim/train_fe_ohe.csv")
	df_test = pd.read_csv("../input/interim/test_fe_ohe.csv")
	df_ss = pd.read_csv("../input/interim/ss_fe_ohe.csv")

	cols = ["Weeks", "Percent", "Age", "First_Week", "First_FVC", "Weeks_Passed", "Height"]

	nz = Normalize(df_train, cols)
	train_fe_ohe_s = nz.scale_features()

	nz = Normalize(df_test, cols)
	test_fe_ohe_s = nz.scale_features()

	nz = Normalize(df_ss, cols)
	ss_fe_ohe_s = nz.scale_features()

	print(train_fe_ohe_s.head())

	train_fe_ohe_s.to_csv("../input/interim/train_fe_ohe_s.csv", index=False)
	test_fe_ohe_s.to_csv("../input/interim/test_fe_ohe_s.csv", index=False)
	ss_fe_ohe_s.to_csv("../input/interim/ss_fe_ohe_s.csv", index=False)
import pandas as pd
import numpy as np

import myutils

class FeatureEngineering:
	def __init__(
		self,
		train=None,
		test=None,
		submission=None,
		):

		self.train = train
		self.test = test
		self.submission = submission

		train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])	
		if self.submission is not None:
			self.submission['Patient'] = (
				self.submission['Patient_Week']
				.apply(
					lambda x: x.split("_")[0]
				)
			)

			self.submission['Weeks'] = (
				self.submission['Patient_Week']
				.apply(
					lambda x: x.split("_")[1]
				)
			)

			self.submission["Weeks"] = self.submission["Weeks"].astype(int)
		

	def linear(self):
		ss = self.submission[['Patient', "Weeks", "Confidence", "Patient_Week"]]
		ss = ss.merge(self.test.drop("Weeks", axis=1), on='Patient')

		# ss.head()
		self.train["Dataset"] = "train"
		self.test["Dataset"] = "test"
		ss["Dataset"] = "ss"

		all_data = self.train.append([self.test, ss])
		all_data = all_data.reset_index(drop=True)

		all_data["First_Week"] = all_data["Weeks"]
		all_data.loc[all_data.Dataset == "ss", "First_Week"] = np.nan
		all_data['First_Week'] = all_data.groupby("Patient")["First_Week"].transform("min")

		first_fvc = (
			all_data
			.loc[all_data.Weeks == all_data.First_Week][["Patient", "FVC"]]
			.rename({"FVC": "First_FVC"}, axis=1)
			.groupby("Patient")
			.first()
			.reset_index()
		)

		all_data = all_data.merge(first_fvc, on="Patient", how="left")
		all_data["Weeks_Passed"] = all_data["Weeks"] - all_data["First_Week"]
		all_data["Height"] = all_data.apply(myutils.calculate_height, axis=1)

		self.train = all_data.loc[all_data.Dataset == 'train']
		self.test = all_data.loc[all_data.Dataset == 'test']
		self.submission = all_data.loc[all_data.Dataset == 'ss']

		return self.train, self.test, self.submission

if __name__ == "__main__":
	df_train = pd.read_csv("../input/raw/train.csv")
	df_test = pd.read_csv("../input/raw/test.csv")
	df_ss = pd.read_csv("../input/raw/sample_submission.csv")

	fe = FeatureEngineering(df_train, df_test, df_ss)

	train_fe, test_fe, ss_fe = fe.linear()
	
	train_fe.to_csv("../input/interim/train_fe.csv", index=False)
	test_fe.to_csv("../input/interim/test_fe.csv", index=False)
	ss_fe.to_csv("../input/interim/ss_fe.csv", index=False) 



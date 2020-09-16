import pandas as pd
from . import utils

class FeatureEngineering:
	def __init(
		self,
		train,
		test=None,
		submission=None,
		):

		self.train = train
		self.test = test
		self.submission = submission

		train.drop_duplicates(keep="first", inplace=True, subset=["Patient", "Weeks"])	
		if self.submission not None:
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
		

	def linear(self):
		ss = self.submission[['Patient', "Weeks", "Confidence", "Patient_Week"]]
		ss = ss.merge(test.drop("Weeks", axis=1), on='Patient')

		# ss.head()
		self.train["Dataset"] = "train"
		self.test["Dataset"] = "test"
		ss["Dataset"] = "ss"

		all_data = self.train.append([self.test, ss])
		all_data = all_data.reset_index(drop=True)

		all_data["First_Week"] = all_data["Weeks"]
		all_data[all_data.Dataset == "ss", "First_Week"] = np.nan
		all_data['First_Week'] = all_data.groupby("Patient").transform("min")

		first_fvc = (
			all_data
			.loc[all_data.Weeks == all_data.First_Week][["Patient", "FVC"]]
			.rename({"FVC": "First_FVC"})
			.groupby("Patient")
			.first()
			.reset_index()
		)

		all_data = all_data.merge(first_fvc, on="Patient", how="left")
		all_data["Weeks_Passed"] = all_data["Weeks"] - all_data["First_Week"]
		all_data["Height"] = all_data.apply(utils.calculate_height(), axis=1)




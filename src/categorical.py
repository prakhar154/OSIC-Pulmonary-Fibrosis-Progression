import pandas as pd
import myutils

class CategoricalFeatures:
	def __init__(self, df, categorical_features, encoding_type, handle_na=False):
		self.df = df
		self.cat_features = categorical_features
		self.enc_type = encoding_type
		self.handle_na = handle_na
		self.label_encoders = dict()
		self.binary_encoders = dict()
		self.ohe = None

		if self.handle_na:
			for c in self.cat_features:
				self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
		self.output_df = self.df.copy(deep=True)

	def _one_hot(self):
		self.output_df = pd.get_dummies(data=self.output_df, columns=self.cat_features)
		# self.output_df.drop(columns=self.cat_features)

		return self.output_df

	def fit_tranform(self):
		if self.enc_type == 'ohe':
			return self._one_hot()

if __name__ == "__main__":
	df_train = pd.read_csv("../input/interim/train_fe.csv")
	df_test = pd.read_csv("../input/interim/test_fe.csv")
	df_ss = pd.read_csv("../input/interim/ss_fe.csv")

	cols = ["Sex", "SmokingStatus"]
	
	cat_feats = CategoricalFeatures(df_train, categorical_features=cols, encoding_type="ohe", handle_na=True)
	train_fe_ohe = cat_feats._one_hot()
	
	cat_feats = CategoricalFeatures(df_test, categorical_features=cols, encoding_type="ohe", handle_na=True)
	test_fe_ohe = cat_feats._one_hot()
	
	cat_feats = CategoricalFeatures(df_ss, categorical_features=cols, encoding_type="ohe", handle_na=True)
	ss_fe_ohe = cat_feats._one_hot()
	

	train_fe_ohe.to_csv("../input/interim/train_fe_ohe.csv", index=False)
	test_fe_ohe.to_csv("../input/interim/test_fe_ohe.csv", index=False)
	ss_fe_ohe.to_csv("../input/interim/ss_fe_ohe.csv", index=False)
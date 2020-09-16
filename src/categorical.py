import pandas as pd

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
	df_train = pd.read_csv("../input/train.csv")
	print(df_train.head())
	cols = ["Sex", "SmokingStatus"]
	cat_feats = CategoricalFeatures(df_train, categorical_features=cols, encoding_type="ohe", handle_na=True)
	data_transformed = cat_feats._one_hot()
	print(data_transformed.head())
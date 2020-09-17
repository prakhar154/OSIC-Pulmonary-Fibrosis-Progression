import pandas as pd
from sklearn.model_selection import GroupKFold

if __name__ == "__main__":
	df = pd.read_csv("../input/interim/train_fe_ohe.csv")
	df["kfold"] = -1
	df = df.sample(frac=1).reset_index(drop=True)
	groups = df['Patient'].values

	gfk = GroupKFold(n_splits=5)

	for fold, (train_idx, val_idx) in enumerate(gfk.split(df, df.FVC.values, groups)):
		print(len(train_idx), len(val_idx))
		df.loc[val_idx, 'kfold'] = fold

	df.to_csv("../input/interim/train_fe_ohe_folds.csv", index=False)

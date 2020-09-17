import pandas as pd
import os

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = os.environ.get("FOLD")
MODEL = os.environ.get("MODEL")

if __name__ == "__main__":
	df = pd.read_csv(TRAINING_DATA)
	df_test = pd.read_csv(TEST_DATA)
	print(df.head())
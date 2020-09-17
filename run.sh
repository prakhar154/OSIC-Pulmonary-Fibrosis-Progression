export TRAINING_DATA=input/interim/train_folds.csv
export TEST_DATA=input/raw/test.csv

export MODEL=$1

python3 -m src.train

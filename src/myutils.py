def calculate_height(row):
	if row["Sex"] == "Male":
		return row["First_FVC"] / (27.63 - .112 * row["Age"])
	else:
		return row["First_FVC"] / (21.78 - .101 * row["Age"])


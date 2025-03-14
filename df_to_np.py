import numpy as np
import pandas as pd

def split_dataset(pizza: np.array, sushi: np.array, shawarma: np.array) -> tuple[np.array, np.array, np.array]:
	# Add an extra column to each category with the corresponding label
	pizza = np.column_stack((pizza, np.full(len(pizza), 0)))
	sushi = np.column_stack((sushi, np.full(len(sushi), 1)))
	shawarma = np.column_stack((shawarma, np.full(len(shawarma), 2)))

	# Shuffle each category
	np.random.shuffle(pizza)
	np.random.shuffle(sushi)
	np.random.shuffle(shawarma)

	# Split each category into training, validation, and test subsets
	def split_data(data):
		train, validate, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])
		return train, validate, test

	pizza_train, pizza_validate, pizza_test = split_data(pizza)
	sushi_train, sushi_validate, sushi_test = split_data(sushi)
	shawarma_train, shawarma_validate, shawarma_test = split_data(shawarma)

	# Combine each of the training subsets, validation subsets, and test subsets
	train_combined = np.concatenate((pizza_train, sushi_train, shawarma_train))
	validate_combined = np.concatenate((pizza_validate, sushi_validate, shawarma_validate))
	test_combined = np.concatenate((pizza_test, sushi_test, shawarma_test))

	# Shuffle each of the combined sets
	np.random.shuffle(train_combined)
	np.random.shuffle(validate_combined)
	np.random.shuffle(test_combined)

	return train_combined, validate_combined, test_combined

def split_X_t(data_subset: np.array) -> tuple[np.array, np.array]:
	return data_subset[:, :-1], data_subset[:, -1].reshape(-1, 1)

def df_to_np(df: pd.datafrme):
    # Separate the dataframe into three based on the 'Label' column
	df_pizza = df[df['Label'] == 'Pizza']
	df_sushi = df[df['Label'] == 'Sushi']
	df_shawarma = df[df['Label'] == 'Shawarma']

	# Drop the 'Label' column from each dataframe
	df_pizza = df_pizza.drop(columns=['Label', 'id'])
	df_sushi = df_sushi.drop(columns=['Label', 'id'])
	df_shawarma = df_shawarma.drop(columns=['Label', 'id'])

	# Convert each dataframe to a numpy array
	np_pizza = df_pizza.to_numpy()
	np_sushi = df_sushi.to_numpy()
	np_shawarma = df_shawarma.to_numpy()

	train_set, validation_set, test_set = split_dataset(np_pizza, np_sushi, np_shawarma)

	train_X, train_t = split_X_t(train_set)
	validation_X, validation_t = split_X_t(validation_set)
	test_X, test_t = split_X_t(test_set)

	return train_X, train_t, validation_X, validation_t, test_X, test_t

if __name__ == '__main__':
    df = pd.read_csv('cleaned_data_combined_modified.csv')
    train_X, train_t, validation_X, validation_t, test_X, test_t = df_to_np(df)
    np.savez(
        'np_cleaned_split_shuffled_data.npz',
        train_X=train_X,
        train_t=train_t,
        validation_X=validation_X,
        validation_t=validation_t,
        test_X=test_X,
        test_t=test_t
        )

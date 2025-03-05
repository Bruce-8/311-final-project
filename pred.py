"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

	from example_pred import predict_all
	predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random
import re

# numpy and pandas are also permitted
import numpy
import pandas as pd

def q1_q8_preprocess(df, col):
	ind_var_cols = pd.get_dummies(df[col], prefix=col)
	ind_var_cols = ind_var_cols.astype(int)
	df = pd.concat([df, ind_var_cols], axis=1)
	df = df.drop(col, axis=1)
	return df

def q3_q7_preprocess(df, col):
	df_split = df[col].str.split(',', expand=True)
	df_split = df_split.apply(pd.Series.value_counts, axis=1).fillna(0).astype(int)
	df = pd.concat([df, df_split], axis=1)
	df = df.drop(col, axis=1)
	return df

numbers_word_representations = {
	"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
	"six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
	"eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
	"sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
	"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
	"eighty": 80, "ninety": 90, "hundred": 100
}

def preprocess_word_representations_to_numbers(text):
	words = text.lower().split()

	converted_words = []
	i = 0

	while i < len(words):
		word = words[i]
		# Compound numbers split by hyphens (ex. "twenty-five")
		if "-" in word and word.count("-") == 1:
			parts = word.split("-")
			if parts[0] in numbers_word_representations and parts[1] in numbers_word_representations:
				num_value = numbers_word_representations[parts[0]] + numbers_word_representations[parts[1]]
				converted_words.append(str(num_value))
				i += 1
				continue
			else:
				converted_words.append(word)

		# Single word numbers
		elif word.rstrip(".") in numbers_word_representations:
			# Check compound numbers not split by hyphen (ex. "twenty five")
			word = word.rstrip(".")
			if i + 1 < len(words) and words[i + 1] in numbers_word_representations:  
				num_value = numbers_word_representations[word] + numbers_word_representations[words[i + 1]]
				converted_words.append(str(num_value))
				i += 2
				continue
			else:
				converted_words.append(str(numbers_word_representations[word]))

		# Already numerical digits
		elif re.match(r'^\d+$', word):
			converted_words.append(word)

		else: # keep everything else
			converted_words.append(word)

		i += 1

	# Merge hyphenated numbers for later regex
	for i, word in enumerate(converted_words):
		if word == "-" and i > 0 and i < len(converted_words) - 1:
			if re.match(r'\$?\d+(?:\.\d+)?\$?', converted_words[i - 1]) and re.match(r'\$?\d+(?:\.\d+)?\$?', converted_words[i + 1]):
				converted_words[i - 1] = f"{converted_words[i - 1]}-{converted_words[i + 1]}"
				del converted_words[i:i+2]
	
	return " ".join(converted_words)

def q2_extract_data(input):
	if not isinstance(input, str):
		return input
	
	processed_input = preprocess_word_representations_to_numbers(input)

	# 1. detect hyphenated range (ex. "7-10") and average them
	#    if there's multiple, choose the largest range
	hyphenated_range_matches = re.findall(r'(\d+)-(\d+)', processed_input)
	if hyphenated_range_matches:
		return max((int(a) + int(b)) / 2 for a, b in hyphenated_range_matches)

	# 2. detect hyphenated range with the word "to" (ex. "7 to 10") and average them
	#    if there's multiple, choose the largest range
	hyphenated_range_to_matches = re.findall(r'(\d+)[ ]+to[ ]+(\d+)', processed_input)
	if hyphenated_range_to_matches:
		return max((int(a) + int(b)) / 2 for a, b in hyphenated_range_to_matches)
	
	# 3. detect hyphenated range with the word "or" (ex. "7 or 8") and average them
	#       if there's multiple, choose the largest range
	hyphenated_range_or_matches = re.findall(r'(\d+)[ ]+or[ ]+(\d+)', processed_input)
	if hyphenated_range_or_matches:
		return max((int(a) + int(b)) / 2 for a, b in hyphenated_range_or_matches)

	# 3. detect numbers, and use maximum number provided
	number_matches = re.findall(r'\d+', processed_input)
	if number_matches:
		return max(map(int, number_matches))
	
	# 4. detect comma separated ingredients
	if "," in processed_input:
		return processed_input.count(",") + 1
	
	# 5. detect bullet pointed ingredients using *
	if "*" in processed_input:
		return processed_input.count("*")

	# 6. detect bullet pointed ingredients with just white space
	ingredients = [ingredient.strip() for ingredient in re.split(r'\n+', input) if ingredient.strip()]
	return len(ingredients)

def q4_extract_data(input):
	if not isinstance(input, str):
		return None 
	
	processed_input = preprocess_word_representations_to_numbers(input)

	# Find optional $ then a number, then optional $ and optional spaces followed by currency (cad, dollar, etc..)
	token_pattern = r'(?:\$)?\d+(?:\.\d+)?(?:\$|(?:\s*(?:cad|(?:canadian\s+)?dollars?)))?'
	def contains_currency(text):
		return bool(re.search(r'(\$|cad|(?:canadian\s+)?dollars?)', text))
	
	# Check if text contains currency markers (indicating this is the provided price)
	currency_present = contains_currency(processed_input)
	
	# Check for hyphenated range patterns
	candidate_ranges = []
	range_patterns = [
		rf'({token_pattern})[-~]({token_pattern})',
		rf'({token_pattern})\s+to\s+({token_pattern})',
		rf'({token_pattern})\s+or\s+({token_pattern})'
	]
	for pattern in range_patterns:
		for match in re.finditer(pattern, processed_input):
			part1, part2 = match.group(1), match.group(2)
			try:
				num1 = float(re.sub(r'[^\d\.]', '', part1))
				num2 = float(re.sub(r'[^\d\.]', '', part2))
			except Exception:
				continue
			average = (num1 + num2) / 2.0
			# Check if either part contains a currency marker
			if contains_currency(part1) or contains_currency(part2):
				with_currency = True
			else:
				with_currency = False
			candidate_ranges.append((average, with_currency))
	
	# Only consider candidate ranges that have currency info
	if currency_present:
		candidate_ranges = [cr for cr in candidate_ranges if cr[1]]
	if candidate_ranges:
		best_range = max(candidate_ranges, key=lambda x: x[0])
		return best_range[0]
	
	# Look for individual numbers 
	candidate_numbers = []
	number_pattern = token_pattern
	for match in re.finditer(number_pattern, processed_input):
		token = match.group(0)
		try:
			value = float(re.sub(r'[^\d\.]', '', token))
		except Exception:
			continue
		token_has_currency = contains_currency(token)
		candidate_numbers.append((value, token_has_currency))
	
	# Only consider candidate numbers that have currency info
	if currency_present:
		candidate_numbers = [cn for cn in candidate_numbers if cn[1]]
	if candidate_numbers:
		best_number = max(candidate_numbers, key=lambda x: x[0])
		return best_number[0]
	
	# No numerical values provided - use sklearn's SimpleImputer to handle this
	return float('nan')

def q2_preprocess(df, col):
	df[col] = df[col].apply(q2_extract_data)
	df = df.rename(columns={col: "ingredient_count"})
	return df

def q4_preprocess(df, col):
	df[col] = df[col].apply(q4_extract_data)
	df = df.rename(columns={col: "expected_price"})
	return df

def remove_words(words: list[str], ting: str):
		for word in words:
			ting = re.sub(r'\b' + word + r'\b', '', ting)
		return ting

def process_string(value: str) -> str:
	# Convert the value to lowercase
	value = str(value).lower()
	# Remove all generic words
	value = remove_words(['the', 'an', 'and', 'of', 'to'], value)
	# Remove all s after a word
	value = re.sub(r's\b', '', value)
	# Remove all characters that are not consonants
	value = re.sub(r'[^bcdfghjklmnpqrstvwxyz]', '', value)
	# Combine data points related to "no movie" to data points related to "none"
	if value == 'nn':
		value = 'nmv'
	return value

def get_big_buckets(df: pd.DataFrame) -> dict:
	tings = {}
	specifics = {}

	def add_to_tings(tingz: dict, ting: str, original: str = None):
		if original:
			if ting in specifics:
				specifics[ting].append(original)
			else:
				specifics[ting] = [original]

		if ting in tingz:
			tingz[ting] += 1
		else:
			tingz[ting] = 1

	def remove_words(words: list[str], ting: str):
		for word in words:
			ting = re.sub(r'\b' + word + r'\b', '', ting)
		return ting

	# Loop through the values of the second column and print each one
	for value in df.iloc[:, 1]:
		add_to_tings(tings, process_string(value), value)

	bigtings = {}
	bigenough = 15
	for ting in tings:
		if tings[ting] > bigenough:
			bigtings[ting] = tings[ting]
	
	return bigtings

def q_5_preprocess(df: pd.DataFrame) -> pd.DataFrame:
	# Drop the columns that are not needed
	buckets = {
		'cldywthchncmtbll',
		'nvm',
		'hmln',
		'spdrmn',
		'tngmtntnnjtrtl',
		'rttll',
		'vngr',
		'fndngnm',
		'rshhr',
		'dcttr',
		'lddn',
		'sprtdwy',
		'jrdrmssh',
		'mnstrnc',
	}
	
	def bucket_map(value: str) -> str:
		value = process_string(value)
		if value in buckets:
			return value
		return ''

	df['big_bucket'] = df.iloc[:, 1].map(bucket_map)
	df = df.drop('Q5: What movie do you think of when thinking of this food item?', axis=1)
	return df

class DrinkClassifier:
	def classify_drink(self, drink_name):
		res = None
		tests = [self.is_water,
				 self.is_fancy_alcohol,
				 self.is_soft_drink,
				 self.is_tea,
				 self.is_juice,
				 self.is_soup
				 ]
		drink_name = str(drink_name)
		for test in tests:
			temp_res = test(drink_name)
			if temp_res:
				res = test.__name__
				return pd.Series([int(func.__name__ == res) for func in tests])

		return pd.Series([0] * len(tests))

	def is_water(self, drink_name):
		return re.search(r'water', drink_name, re.IGNORECASE) is not None

	def is_fancy_alcohol(self, drink_name):
		regex_str = r'(sake|wine|beer|cocktail|whiskey|vodka|rum|gin|tequila|margarita|martini)'
		return re.search(regex_str, drink_name, re.IGNORECASE) is not None

	# includes iced tea
	def is_soft_drink(self, drink_name):
		regex_str = r'(coke|soda|sprite|pepsi|cola|ale|pop|diet|ice|nestea|lemon)'
		return re.search(regex_str, drink_name, re.IGNORECASE) is not None

	def is_tea(self, drink_name):
		regex_str = r'tea|chai|oolong'
		return re.search(regex_str, drink_name, re.IGNORECASE) is not None

	def is_juice(self, drink_name):
		regex_str = r'juice'
		return re.search(regex_str, drink_name, re.IGNORECASE) is not None

	def is_soup(self, drink_name):
		regex_str = r'soup|ramen|pho|chowder|broth|bisque|stew'
		return re.search(regex_str, drink_name, re.IGNORECASE) is not None

def q6_preprocess(df, col):
	df[['water',
	  'alcohol',
	  'soft_drink',
	  'tea',
	  'juice',
	  'soup']] = df[col].apply(DrinkClassifier().classify_drink)
	df = df.drop(col, axis=1)
	return df

def preprocess(filename, output=False):
	df = pd.read_csv(filename)
	df = q1_q8_preprocess(df, "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)")
	df = q1_q8_preprocess(df, "Q8: How much hot sauce would you add to this food item?")
	df = q3_q7_preprocess(df, "Q3: In what setting would you expect this food to be served? Please check all that apply")
	df = q3_q7_preprocess(df, "Q7: When you think about this food item, who does it remind you of?")
	df = q2_preprocess(df, "Q2: How many ingredients would you expect this food item to contain?")
	df = q4_preprocess(df, 'Q4: How much would you expect to pay for one serving of this food item?')
	df = q_5_preprocess(df)
	df = q6_preprocess(df, 'Q6: What drink would you pair with this food item?')

	if output:
		df.to_csv('output.csv', index=False)
	
	return df

def predict(x):
	pass

def predict_all(filename):
	"""
	Make predictions for the data in filename
	"""
	df = pd.read_csv(filename)
	df = preprocess(df, output=False)

	predictions = []
	for test_example in df:
		# obtain a prediction for this test example
		pred = predict(test_example)
		predictions.append(pred)

	return predictions

if __name__ == "__main__":
	preprocess("cleaned_data_combined_modified.csv", output=True)

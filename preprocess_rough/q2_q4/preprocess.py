import pandas as pd
import re

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
    return df

def q4_preprocess(df, col):
    df[col] = df[col].apply(q4_extract_data)
    return df

if __name__ == "__main__":
    df = pd.read_csv("cleaned_data_combined_modified.csv")
    df = q2_preprocess(df, 'Q2: How many ingredients would you expect this food item to contain?')
    df = q4_preprocess(df, 'Q4: How much would you expect to pay for one serving of this food item?')
    df.to_csv('output.csv', index=False)

    # Compare for Q2
    df1 = pd.read_csv("cleaned_data_combined_modified.csv", usecols=["id", "Q2: How many ingredients would you expect this food item to contain?"])
    df2 = pd.read_csv("output.csv", usecols=["Q2: How many ingredients would you expect this food item to contain?"])
    df1.rename(columns={"Q2: How many ingredients would you expect this food item to contain?": "Q2_original"}, inplace=True)
    df2.rename(columns={"Q2: How many ingredients would you expect this food item to contain?": "Q2_extracted"}, inplace=True)
    df_combined = pd.concat([df1, df2], axis=1)
    df_combined.to_csv("q2_test.csv", index=False)

    # Compare for Q4
    df1 = pd.read_csv("cleaned_data_combined_modified.csv", usecols=["id", "Q4: How much would you expect to pay for one serving of this food item?"])
    df2 = pd.read_csv("output.csv", usecols=["Q4: How much would you expect to pay for one serving of this food item?"])
    df1.rename(columns={"Q4: How much would you expect to pay for one serving of this food item?": "Q4_original"}, inplace=True)
    df2.rename(columns={"Q4: How much would you expect to pay for one serving of this food item?": "Q4_extracted"}, inplace=True)
    df_combined = pd.concat([df1, df2], axis=1)
    df_combined.to_csv("q4_test.csv", index=False)
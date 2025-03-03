"""
This module preprocesses Q1, Q3, and Q7 columns
of the CSC311 Project Data CSV File.

Prompts (ChatGPT):
- "how to convert a column into indicator variables"
- "if i have a column with information like 'today, tomorrow' as
   a sample entry and i want to convert this to today and tomorrow as individual
   indicator variables"

"""
import pandas as pd

def convert_to_indicator(df, col):
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


if __name__ == '__main__':
    df = pd.read_csv('cleaned_data_combined_modified.csv')
    df = convert_to_indicator(df, "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)")
    df = convert_to_indicator(df, "Q8: How much hot sauce would you add to this food item?")
    df = q3_q7_preprocess(df, "Q3: In what setting would you expect this food to be served? Please check all that apply")
    df = q3_q7_preprocess(df, "Q7: When you think about this food item, who does it remind you of?")
    df.to_csv('sample_output.csv', index=False)
    print(df)
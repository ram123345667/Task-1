!pip install word2number
import pandas as pd
import numpy as np  # Import numpy for np.nan
from sklearn.impute import SimpleImputer
from word2number import w2n # Import word_to_num

# Load the dataset
df = pd.read_csv('/content/mall_customer_segmentation_dirty_100.csv')

# 1. Remove duplicates
df = df.drop_duplicates()

# 2. Rename columns for uniformity
df.columns = [col.strip().capitalize().replace(' ', '_') for col in df.columns]
print("\nColumn names after renaming:", df.columns.tolist())

# 3. Normalize 'Gender' column
def clean_gender(value):
    val = str(value).strip().lower()
    if val in ['m', 'male']:
        return 'Male'
    elif val in ['f', 'female']:
        return 'Female'
    else:
        return 'Other'

if 'Gender' in df.columns:
    df.loc[:, 'Gender'] = df['Gender'].apply(clean_gender)

# 4. Convert word numbers to integers in numeric columns
def convert_to_number(val):
    try:
        if isinstance(val, str):
            val = val.strip()
            if val.isdigit():
                return int(val)
            # Use word_to_num to convert words to numbers
            return w2n.word_to_num(val.lower())
        # Convert valid integers/floats directly
        # Ensure conversion handles various numeric formats and returns float for potential np.nan
        return float(val)
    except:
        # Return np.nan for values that cannot be converted
        # SimpleImputer works reliably with np.nan
        return np.nan

# Apply conversion to columns expected to be numeric
# This includes 'Spending_Score'
numeric_cols_to_convert = ['Age', 'Spending_Score', 'Annual_Income']
for col in numeric_cols_to_convert:
    if col in df.columns:
        # Apply conversion and explicitly convert to float to handle NaNs before imputation
        # This ensures these columns are numeric types even if they contain NaNs
        df.loc[:, col] = df[col].apply(convert_to_number).astype(float)


# 5. Handle missing values using the appropriate imputer
# Identify numeric and categorical columns *after* type conversions
# numeric_cols will now include 'Age', 'Spending_Score', 'Annual_Income' if conversion to float was successful
numeric_cols = df.select_dtypes(include=np.number).columns
# categorical_cols should now only include 'Gender' if other columns were converted
categorical_cols = df.select_dtypes(include=['object']).columns

# Numeric imputation
if len(numeric_cols) > 0:
    imputer_num = SimpleImputer(strategy='mean')
    # Apply numeric imputation to numeric columns
    df.loc[:, numeric_cols] = imputer_num.fit_transform(df[numeric_cols])

# Categorical imputation (should only apply to actual categorical columns like Gender)
if len(categorical_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    # Apply categorical imputation to categorical columns
    df.loc[:, categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])


# 6. Fix data types (convert numeric columns to appropriate types after imputation)
# Age can often be Int64 if no decimals were introduced by imputation
for col in numeric_cols: # Iterate through columns identified as numeric *after* imputation
    if col in df.columns: # Ensure column still exists
         if col == 'Age':
            try:
                # Convert Age to Int64, allowing for NaN from imputation
                df.loc[:, col] = df[col].astype('Int64')
            except Exception as e:
                print(f"Could not convert Age column to Int64: {e}")
                # Keep as float if conversion fails
                pass # Keep it as float if conversion to Int64 fails
         else:
            # For other numeric columns, you can choose a type (e.g., float64 or Int64)
            try:
                df.loc[:, col] = df[col].astype('float64')
            except Exception as e:
                 print(f"Could not convert column {col} to float64: {e}")
                 pass # Keep original type if conversion fails


# 7. Outlier removal using IQR
# Re-select numeric columns after type conversion for outlier detection
numeric_cols_for_outliers = df.select_dtypes(include=np.number).columns

for col in numeric_cols_for_outliers:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter the DataFrame and use .copy() to avoid SettingWithCopyWarning
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()

# 8. Final info check
if 'Gender' in df.columns:
    print("\nGender value counts after cleaning:")
    print(df['Gender'].value_counts())
else:
    print("\n'Gender' column not found after processing.")

print("\nFinal Cleaned Data Info:")
print(df.info())

# 9. Save cleaned data
df.to_csv('mall_customer_cleaned.csv', index=False)

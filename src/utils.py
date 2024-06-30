import pandas as pd
import re

def clean_text_preserve_apostrophes(text, lowercase=True):
    """
    Clean the text while preserving apostrophes, and adjust the character offsets for a keyword.

    Args:
    - text (str): The input text to clean.
    - lowercase (bool): Whether to convert the text to lowercase. Default is True.

    Returns:
    - cleaned_text (str): The cleaned text.
    """
    if lowercase:
        text = text.lower()

    # Add space around punctuation marks that are not sentiment-expressing and not apostrophes
    text = re.sub(r'(?<=[\w])([,:;])(?=[\w])', r' \1 ', text)

    # Remove HTML tags if any
    text = re.sub(r'<.*?>', '', text)

    # Keep letters, numbers, spaces, !, ?, ., and apostrophes; remove other special characters
    text = re.sub(r"[^a-zA-Z0-9\s!?\.']", '', text)

    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def adjust_keyword_and_offsets(row):
    # Assuming clean_text is your text cleaning function
    cleaned_text = clean_text_preserve_apostrophes(row['Text'])
    cleaned_keyword = clean_text_preserve_apostrophes(row['Keyword'])  # Clean the keyword in the same way as the text

    try:
        # Find the start position of the cleaned keyword in the cleaned text
        new_start = cleaned_text.index(cleaned_keyword)
        new_end = new_start + len(cleaned_keyword)
        adjusted_position = f"{new_start}:{new_end}"
    except ValueError:
        # If the cleaned keyword isn't found, indicate as such
        adjusted_position = "Not found"

    # Update the row with the new values
    return pd.Series([cleaned_text, cleaned_keyword, adjusted_position], index=['Cleaned_Text', 'Cleaned_Keyword', 'Adjusted_Position'])


# Extracting the keywords based on the adjusted positions and adding them to a new column
def extract_new_keyword(row):
    if row['Adjusted_Position'] != "Not found":
        start, end = map(int, row['Adjusted_Position'].split(':'))
        return row['Cleaned_Text'][start:end]
    return "Keyword not found"

def preprocess_data(data):
  # Apply the improved cleaning function to the 'Text' column, preserving apostrophes
  data['Cleaned_Text'] = data['Text'].apply(lambda x: clean_text_preserve_apostrophes(x, lowercase=True))

  # Apply the function to each row and update the dataframe
  data[['Cleaned_Text', 'Cleaned_Keyword', 'Adjusted_Position']] = data.apply(adjust_keyword_and_offsets, axis=1)

  # Apply the function to each row
  data['New_Keyword'] = data.apply(extract_new_keyword, axis=1)

  data['Cleaned_Keyword'] = data['Cleaned_Keyword'].str.lower()
  data['New_Keyword'] = data['New_Keyword'].str.lower()

  data.drop(columns=['Keyword', 'Position', 'Text', 'Cleaned_Keyword'],inplace = True)
  data.rename(columns={
    'Cleaned_Text': 'Text',
    'Adjusted_Position': 'Position',
    'New_Keyword': 'Keyword'
    }, inplace = True)

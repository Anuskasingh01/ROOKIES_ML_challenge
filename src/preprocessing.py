import pandas as pd
import os
from typing import Optional

def load_tsv(path: str, usecols: Optional[list] = None) -> pd.DataFrame:
    """
    Load a TSV file into a pandas DataFrame.
    
    Args:
        path (str): The file path to the TSV.
        usecols (list, optional): Subset of columns to load.
        
    Returns:
        pd.DataFrame: Loaded DataFrame with string typing and handled missing values.
    """
    return pd.read_csv(path, sep='\t', dtype=str, usecols=usecols, keep_default_na=False)

def normalize_name(series: pd.Series) -> pd.Series:
    """
    Normalize the business_name column.
    
    Args:
        series (pd.Series): The original business names.
        
    Returns:
        pd.Series: The normalized business names.
    """
    # 1. Unicode normalize (NFKC)
    s = series.str.normalize('NFKC')
    
    # 2. Case-fold / lowercase
    s = s.str.lower()
    
    # 3. Convert '&' to 'and'
    s = s.str.replace(r'&', ' and ', regex=True)
    
    # 4. Remove unnecessary punctuation (keep alphanumeric and spaces)
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    
    # 5. Normalize whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return s

def normalize_address(series: pd.Series) -> pd.Series:
    """
    Normalize the business_address column safely handling missing/empty addresses.
    
    Args:
        series (pd.Series): The original business addresses.
        
    Returns:
        pd.Series: The normalized business addresses.
    """
    # Handle missing values by converting NaN to empty string if any
    s = series.fillna('')
    
    # Unicode normalize
    s = s.str.normalize('NFKC')
    
    # Lowercase
    s = s.str.lower()
    
    # Replace punctuation with space to preserve tokens and numbers
    s = s.str.replace(r'[^\w\s]', ' ', regex=True)
    
    # Normalize whitespace
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return s

def normalize_country(series: pd.Series) -> pd.Series:
    """
    Normalize the country column.
    
    Args:
        series (pd.Series): The original country names.
        
    Returns:
        pd.Series: The normalized country names.
    """
    s = series.fillna('')
    
    # Unicode normalize
    s = s.str.normalize('NFKC')
    
    # Lowercase and trim whitespace
    s = s.str.lower().str.strip()
    
    return s

def extract_postal_code(series: pd.Series) -> pd.Series:
    """
    Extract postal code (Indian 6-digit PIN code, US 5-digit ZIP, or ZIP+4) from business addresses.
    
    Args:
        series (pd.Series): The business addresses.
        
    Returns:
        pd.Series: Extracted postal codes, or empty string if not found.
    """
    s = series.fillna('').astype(str)
    pattern = r'\b(\d{6}|\d{5}(?:-\d{4})?)\b'
    extracted = s.str.extract(pattern, expand=False).fillna('')
    return extracted

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized columns to the DataFrame rather than replacing originals.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        pd.DataFrame: The DataFrame with appended normalized columns:
                      - business_name_normalized & normalized_name
                      - business_address_normalized & normalized_address
                      - country_normalized
                      - postal_code
    """
    df = df.copy()
    if 'business_name' in df.columns:
        norm_name = normalize_name(df['business_name'])
        df['business_name_normalized'] = norm_name
        df['normalized_name'] = norm_name
    if 'business_address' in df.columns:
        norm_addr = normalize_address(df['business_address'])
        df['business_address_normalized'] = norm_addr
        df['normalized_address'] = norm_addr
        df['postal_code'] = extract_postal_code(df['business_address'])
    if 'country' in df.columns:
        df['country_normalized'] = normalize_country(df['country'])
    return df

def normalize_file(input_path: str, output_path: str, chunksize: int = 100000):
    """
    Process a TSV file in chunks to handle millions of rows memory efficiently.
    
    Args:
        input_path (str): The input TSV file path.
        output_path (str): The output TSV file path.
        chunksize (int): The number of rows to process at a time.
    """
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    first_chunk = True
    
    for chunk in pd.read_csv(input_path, sep='\t', chunksize=chunksize, dtype=str, keep_default_na=False):
        normalized_chunk = normalize_dataframe(chunk)
        normalized_chunk.to_csv(
            output_path, 
            sep='\t', 
            index=False, 
            mode='w' if first_chunk else 'a', 
            header=first_chunk
        )
        first_chunk = False

def validate_schema(df: pd.DataFrame, expected_source: Optional[str] = None):
    """
    Validate that the DataFrame contains the required core columns and optionally check the entity_id prefix.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        expected_source (str, optional): The expected source prefix (S1, S2, or S3).
    """
    required_cols = {'entity_id', 'business_name', 'business_address', 'country'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
        
    if expected_source is not None:
        if expected_source not in {'S1', 'S2', 'S3'}:
            raise ValueError(f"Invalid expected_source: {expected_source}. Must be one of S1, S2, S3")
            
        if len(df) > 0:
            prefix = expected_source + '-'
            if not df['entity_id'].astype(str).str.startswith(prefix).all():
                raise ValueError(f"Not all entity_id values start with expected prefix {prefix}")

def tokenize_name(series: pd.Series) -> pd.Series:
    """
    Tokenize the normalized business name.
    
    Args:
        series (pd.Series): The normalized business names.
        
    Returns:
        pd.Series: A series of lists containing tokens.
    """
    return series.fillna('').str.split()

def extract_address_numbers(series: pd.Series) -> pd.Series:
    """
    Extract numeric tokens from the normalized address.
    
    Args:
        series (pd.Series): The normalized business addresses.
        
    Returns:
        pd.Series: A series of lists containing numeric tokens.
    """
    # Use findall to safely extract all contiguous digit sequences
    return series.fillna('').str.findall(r'\d+')

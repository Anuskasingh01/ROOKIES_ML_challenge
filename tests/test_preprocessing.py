import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import normalize_dataframe, normalize_file

def test_normalization():
    # Small in-memory DataFrame
    data = {
        'entity_id': ['1', '2', '3', '4', '5', '6'],
        'business_name': [
            'Café Mocha',           # Unicode
            'STARBUCKS CORP.',      # Uppercase, punctuation
            'Barnes & Noble',       # Ampersand
            '  Space   Bar  ',      # Whitespace
            'T.J. Maxx, Inc!',      # Punctuation
            'Täco Bęll'             # Non-ASCII that should NOT be deleted
        ],
        'business_address': [
            '123 4th St. Apt 5',    # Numbers and punctuation
            '',                     # Empty address
            None,                   # Missing address
            'P.O. Box 999, NY',     # Punctuation
            '10-12 Main St   ',     # Whitespace
            '123 Møcha Gata'        # Non-ASCII that should NOT be deleted
        ],
        'country': [
            ' US ',
            'India',
            'France',
            'UNKNOWN',
            '  us',
            'Deutschland'
        ]
    }
    
    df = pd.DataFrame(data)
    # Simulate pandas reading TSV where missing values might be NaN
    
    norm_df = normalize_dataframe(df)
    
    # 1. Preservation of original columns
    assert 'entity_id' in norm_df.columns
    assert 'business_name' in norm_df.columns
    assert 'business_address' in norm_df.columns
    assert 'country' in norm_df.columns
    
    pd.testing.assert_series_equal(norm_df['entity_id'], df['entity_id'])
    pd.testing.assert_series_equal(norm_df['business_name'], df['business_name'])
    pd.testing.assert_series_equal(norm_df['business_address'], df['business_address'])
    
    # 2. Name Normalization
    assert norm_df['business_name_normalized'].iloc[0] == 'café mocha' # Unicode handled and kept
    assert norm_df['business_name_normalized'].iloc[1] == 'starbucks corp' # Lowercase, no punctuation
    assert norm_df['business_name_normalized'].iloc[2] == 'barnes and noble' # Ampersand replaced
    assert norm_df['business_name_normalized'].iloc[3] == 'space bar' # Whitespace trimmed
    assert norm_df['business_name_normalized'].iloc[4] == 't j maxx inc' # Punctuation replaced with spaces
    assert norm_df['business_name_normalized'].iloc[5] == 'täco bęll' # Non-ASCII preserved
    
    # 3. Address Normalization
    assert norm_df['business_address_normalized'].iloc[0] == '123 4th st apt 5' # Kept numbers
    assert norm_df['business_address_normalized'].iloc[1] == '' # Empty handled safely
    assert norm_df['business_address_normalized'].iloc[2] == '' # None handled safely
    assert norm_df['business_address_normalized'].iloc[3] == 'p o box 999 ny'
    assert norm_df['business_address_normalized'].iloc[4] == '10 12 main st'
    assert norm_df['business_address_normalized'].iloc[5] == '123 møcha gata' # Non-ASCII preserved
    
    # 4. Country Normalization
    assert norm_df['country_normalized'].iloc[0] == 'us'
    assert norm_df['country_normalized'].iloc[1] == 'india'
    assert norm_df['country_normalized'].iloc[2] == 'france'
    assert norm_df['country_normalized'].iloc[3] == 'unknown' # Preserved unknown
    assert norm_df['country_normalized'].iloc[4] == 'us'
    
    # 5. Deterministic
    norm_df2 = normalize_dataframe(df)
    pd.testing.assert_frame_equal(norm_df, norm_df2)

def test_chunked_vs_full_equivalence(tmp_path):
    data = {
        'entity_id': ['1', '2', '3', '4'],
        'business_name': ['A B', 'C D', 'E F', 'G H'],
        'business_address': ['St 1', 'St 2', 'St 3', 'St 4'],
        'country': ['US', 'IN', 'FR', 'UK']
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "input.tsv"
    output_file = tmp_path / "output.tsv"
    
    df.to_csv(input_file, sep='\t', index=False)
    
    # Full processing
    df_loaded = pd.read_csv(input_file, sep='\t', dtype=str, keep_default_na=False)
    df_full_norm = normalize_dataframe(df_loaded)
    
    # Chunked processing
    normalize_file(str(input_file), str(output_file), chunksize=2)
    df_chunk_norm = pd.read_csv(output_file, sep='\t', dtype=str, keep_default_na=False)
    
    pd.testing.assert_frame_equal(df_full_norm, df_chunk_norm)

def test_validate_schema():
    from src.preprocessing import validate_schema
    
    df = pd.DataFrame({
        'entity_id': ['S1-123'],
        'business_name': ['A'],
        'business_address': ['B'],
        'country': ['C']
    })
    
    # Valid
    validate_schema(df, 'S1')
    validate_schema(df)
    
    # Missing col
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df.drop(columns=['country']))
        
    # Invalid prefix
    with pytest.raises(ValueError, match="Not all entity_id values start with expected prefix S2-"):
        validate_schema(df, 'S2')
        
    # Invalid expected source
    with pytest.raises(ValueError, match="Invalid expected_source"):
        validate_schema(df, 'S4')
        
    # Empty dataframe
    validate_schema(pd.DataFrame(columns=['entity_id', 'business_name', 'business_address', 'country']), 'S1')

def test_helpers():
    from src.preprocessing import tokenize_name, extract_address_numbers
    
    # Names
    names = pd.Series(['raj investments llp', 'one', '', '   '])
    tokens = tokenize_name(names)
    assert list(tokens.iloc[0]) == ['raj', 'investments', 'llp']
    assert list(tokens.iloc[1]) == ['one']
    assert list(tokens.iloc[2]) == []
    assert list(tokens.iloc[3]) == []
    
    # Addresses
    addresses = pd.Series(['85 wayne avenue ticonderoga ny', '3315 fremont street peoria il', 'no numbers here', 'apt 5 and 6'])
    nums = extract_address_numbers(addresses)
    assert list(nums.iloc[0]) == ['85']
    assert list(nums.iloc[1]) == ['3315']
    assert list(nums.iloc[2]) == []
    assert list(nums.iloc[3]) == ['5', '6']

if __name__ == '__main__':
    pytest.main(['-v', __file__])

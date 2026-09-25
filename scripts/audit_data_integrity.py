import pandas as pd
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from src.preprocessing import validate_schema

def audit_file(filepath, expected_source):
    print(f"Auditing {os.path.basename(filepath)}...")
    chunksize = 250000
    
    total_rows = 0
    columns = []
    
    missing_counts = {}
    empty_counts = {}
    
    seen_ids = set()
    dup_ids = 0
    
    seen_rows = set()
    dup_rows = 0
    
    invalid_prefixes = 0
    
    country_dist = {}
    
    min_name_len = float('inf')
    max_name_len = 0
    min_addr_len = float('inf')
    max_addr_len = 0
    
    schema_valid = True
    schema_error = ""

    first = True
    
    for chunk in pd.read_csv(filepath, sep='\t', dtype=str, keep_default_na=False, chunksize=chunksize):
        if first:
            columns = list(chunk.columns)
            try:
                validate_schema(chunk, expected_source)
            except ValueError as e:
                schema_valid = False
                schema_error = str(e)
            for c in columns:
                missing_counts[c] = 0
                empty_counts[c] = 0
            first = False
            
        total_rows += len(chunk)
        
        for c in columns:
            missing_counts[c] += chunk[c].isna().sum()
            empty_counts[c] += (chunk[c] == '').sum()
            
        if 'entity_id' in chunk.columns:
            for eid in chunk['entity_id']:
                if eid in seen_ids:
                    dup_ids += 1
                else:
                    seen_ids.add(eid)
                if expected_source and not str(eid).startswith(expected_source + '-'):
                    invalid_prefixes += 1
                    
        for row in chunk.itertuples(index=False):
            r_str = "\t".join(str(x) for x in row)
            if r_str in seen_rows:
                dup_rows += 1
            else:
                seen_rows.add(r_str)
                
        if 'country' in chunk.columns:
            for c in chunk['country']:
                country_dist[c] = country_dist.get(c, 0) + 1
                
        if 'business_name' in chunk.columns:
            lens = chunk['business_name'].str.len()
            min_name_len = min(min_name_len, lens.min())
            max_name_len = max(max_name_len, lens.max())
            
        if 'business_address' in chunk.columns:
            lens = chunk['business_address'].str.len()
            min_addr_len = min(min_addr_len, lens.min())
            max_addr_len = max(max_addr_len, lens.max())
            
    return {
        'file': os.path.basename(filepath),
        'total_rows': total_rows,
        'columns': columns,
        'missing_counts': missing_counts,
        'empty_counts': empty_counts,
        'dup_ids': dup_ids,
        'dup_rows': dup_rows,
        'invalid_prefixes': invalid_prefixes,
        'country_dist': country_dist,
        'min_name_len': min_name_len,
        'max_name_len': max_name_len,
        'min_addr_len': min_addr_len,
        'max_addr_len': max_addr_len,
        'schema_valid': schema_valid,
        'schema_error': schema_error
    }

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = os.path.join(repo_root, "dataset")
    
    files = [
        (os.path.join(base_dir, 'train', 'train_source1.tsv'), 'S1'),
        (os.path.join(base_dir, 'train', 'train_source2.tsv'), 'S2'),
        (os.path.join(base_dir, 'train', 'train_source3.tsv'), 'S3'),
        (os.path.join(base_dir, 'test', 'test_source1.tsv'), 'S1'),
        (os.path.join(base_dir, 'test', 'test_source2.tsv'), 'S2'),
        (os.path.join(base_dir, 'test', 'test_source3.tsv'), 'S3'),
    ]
    
    reports = []
    
    for fp, src in files:
        if os.path.exists(fp):
            reports.append(audit_file(fp, src))
            
    out_lines = []
    for r in reports:
        out_lines.append(f"=== {r['file']} ===")
        out_lines.append(f"Schema Valid: {r['schema_valid']} {r['schema_error']}")
        out_lines.append(f"Total rows: {r['total_rows']}")
        out_lines.append(f"Columns: {r['columns']}")
        out_lines.append("Missing Counts: " + str(r['missing_counts']))
        out_lines.append("Empty Counts: " + str(r['empty_counts']))
        out_lines.append(f"Duplicate Entity IDs: {r['dup_ids']}")
        out_lines.append(f"Duplicate Rows: {r['dup_rows']}")
        out_lines.append(f"Invalid Prefixes: {r['invalid_prefixes']}")
        out_lines.append(f"Country Distribution: {r['country_dist']}")
        out_lines.append(f"Name Lengths: min={r['min_name_len']}, max={r['max_name_len']}")
        out_lines.append(f"Address Lengths: min={r['min_addr_len']}, max={r['max_addr_len']}")
        out_lines.append("\n")
        
        print(f"{r['file']}: {r['total_rows']} rows, Valid: {r['schema_valid']}, Dup IDs: {r['dup_ids']}, Invalid Prefix: {r['invalid_prefixes']}")

    out_report_dir = os.path.join(repo_root, "reports")
    os.makedirs(out_report_dir, exist_ok=True)
    with open(os.path.join(out_report_dir, "data_integrity_full_audit.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

if __name__ == '__main__':
    main()

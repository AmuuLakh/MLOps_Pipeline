import pandas as pd
import os
import csv
import logging

# configure logger to console and file
logger = logging.getLogger('data_loader')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler('data_load.log', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)


def load_data(path=None):
    """
    Load CSV data from the specified path.
    
    Args:
        path: Path to the CSV file. If None, defaults to data/dataset.csv
        
    Returns:
        pandas DataFrame if successful, None otherwise
    """
    # Only use default path if no path is provided
    if path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, 'data', 'dataset.csv')
    
    # Now use the path parameter (either provided or default)
    if not os.path.exists(path):
        logger.warning(f"File not found: {path}")
        return None
    
    # Check for empty file
    if os.path.getsize(path) == 0:
        logger.warning(f"No data: {path} is empty.")
        return None
    
    try:
        # Try to detect delimiter automatically using csv.Sniffer first
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(4096)
        
        # Only use sniffer if file has content
        if sample.strip():
            try:
                dialect = csv.Sniffer().sniff(sample)
                sep = dialect.delimiter
            except csv.Error:
                # If sniffer fails, default to comma
                sep = ','
        else:
            sep = ','
        
        # Read with detected delimiter
        df = pd.read_csv(path, sep=sep)
        return df
        
    except pd.errors.EmptyDataError:
        logger.warning(f"No data: {path} is empty.")
        return None
    except pd.errors.ParserError:
        # Fallback: try with python engine
        try:
            df = pd.read_csv(path, sep=None, engine='python', on_bad_lines='skip')
            return df
        except Exception:
            logger.error(f"Could not parse CSV (ParserError) for file: {path}")
            return None
    except UnicodeDecodeError:
        # try a fallback encoding
        try:
            df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
            return df
        except Exception:
            logger.error(f"Encoding error reading file: {path}")
            return None
    except Exception as e:
        logger.error(f"Unexpected error loading {path}: {e}")
        return None


if __name__ == "__main__":
    data = load_data()
    if data is None:
        logger.error("Data not loaded.")
    else:
        rows, cols = data.shape
        logger.info("Loaded dataset — rows=%d, cols=%d", rows, cols)
        logger.info("Columns: %s", list(data.columns))
        print(data.head())
'''
This file is part of QuantifierComplexity.
'''
import argparse
import os
from pathlib import Path
import dotenv
import pandas as pd
import numpy as np
import scipy.stats as stats
import utils


# Load environment variables from .env (which is in same dir as src).
# Don't forget to set "PROJECT_DIR" in .env to the name of the location 
# from which you are running current source code.
dotenv.load_dotenv(dotenv.find_dotenv())
# Set paths to relevant directories.
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
RESULTS_DIR = PROJECT_DIR / RESULTS_DIR_RELATIVE

def parse_args():
    '''Create argument parser, add arguments, return parsed args.'''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_expr_len",
        "-l",
        type=int,
        default=MAX_EXPR_LEN,
        help="Generate expressions up to this length",
    )
    parser.add_argument(
        "--max_model_size",
        "-m",
        type=int,
        default=MAX_MODEL_SIZE,
        help="Models up to this size will be used to evaluate the meaning " + \
            "of statements",
    )
    parser.add_argument(
        "--language_name",
        "-j",
        type=str,
        default=LANGUAGE_NAME,
        help="Name of json file (when adding .sjon) that specifies " + \
            "settings",
    )
    parser.add_argument(
        "--dest_dir",
        "-d",
        type=str,
        default=RESULTS_DIR,
        help="Dir to write results to",
    )
    parser.add_argument(
        "--lang_gen_date",
        "-g",
        type=str,
        default=LANG_GEN_DATE,
        help="Date of language generation. Used to load the right csv file. ",
    )
    parser.add_argument(
        "--csv_date",
        "-c",
        type=str,
        default=CSV_DATE,
        help="Date of csv file creation. Used to load the right csv file. ",
    )
    return parser.parse_args()


def shuff_and_standardize_ml(data: pd.DataFrame, verbose=False):
    '''Add zcores and randomly shuffle ml scores.

    "ml" := minimal expression length. Add 3 columns to dataframe: 
    Column "ml_shuffled": randomly shuffled values of expr_length.
    Column "ml_zscore": zscores of expr_length (normalization).
    Column "ml_shuff_zscore": zscores of the shuffled expr_length.

    Args: 
        data: A pandas DataFrame with language data. 
        verbose: True or False. When True, print old and new
            columns, to check.

    '''
    if verbose:
        print(data[["expression", "expr_length"]])
        print("==========================================")
    data["ml_shuffled"] = \
        data["expr_length"].sample(frac=1).reset_index(drop=True)
    if verbose:
        print(data[["expression", "expr_length", "ml_shuffled"]])
    data["ml_zscore"] = stats.zscore(data["expr_length"])
    data["ml_shuff_zscore"] = stats.zscore(data["ml_shuffled"])
    if verbose:
        print(data[["ml_shuffled", "ml_shuff_zscore"]])


def shuff_and_standardize_lz(data: pd.DataFrame, verbose=False):
    '''Add zcores and randomly shuffled lz scores.

    "lz" := minimal expression length. Add 3 columns to dataframe: 
    Column "lz_shuffled": randomly shuffled values of lempel_ziv.
    Column "lz_zscore": zscores of lempel_ziv (normalization).
    Column "lz_shuff_zscore": zscores of the shuffled lempel_ziv.

    Args: 
        data: A pandas DataFrame with language data. 
        verbose: True or False. When True, print old and new
            columns, to check.

    '''
    if verbose:
        print(data[["expression", "lempel_ziv"]])
        print("==========================================")
    data["lz_shuffled"] = \
        data["lempel_ziv"].sample(frac=1).reset_index(drop=True)
    if verbose:
        print(data[["expression", "lempel_ziv", "lz_shuffled"]])
    data["lz_zscore"] = stats.zscore(data["lempel_ziv"])
    data["lz_shuff_zscore"] = stats.zscore(data["lz_shuffled"])
    if verbose:
        print(data[["lz_shuffled", "lz_shuff_zscore"]])


def mon_quan_cons(row):
    '''Return combined property value: 1 iff all props are 1.

    Args: row: A row from pandas dataframe with language data.

    '''
    if (row["monotonicity"] == 1) & (row["quantity"] == 1) \
        & (row["conservativity"] == 1):
        return 1
    else:
        return 0


if __name__ == "__main__":
    # Default values for argparse args.
    LANGUAGE_NAME = "Logical_index"
    MAX_EXPR_LEN = 5
    MAX_MODEL_SIZE = 8
    LANG_GEN_DATE = "2020-12-25"
    CSV_DATE = "2021-01-16"
    args = parse_args()

    # Set DataFrame print options.
    # pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    # pd.set_option("display.max_colwidth", None)

    data = utils.load_csv_data(
        args.max_model_size, args.max_expr_len, args.language_name, 
        args.lang_gen_date, args.csv_date
    )
    
    # Compute binary properties based on graded properties.
    for prop in ["monotonicity", "quantity", "conservativity"]:
        # Rename column name of graded scores: prop --> g_prop.
        data.rename(columns={prop:'g_' + prop[0:4]}, inplace=True)
        # Add column with binary scores under original name: prop.
        # Prop == 1 iff g_prop == 1.0, and prop == 0 otherwise.
        data[prop] = np.where(data['g_' + prop[0:4]] == 1.0, 1, 0)

    shuff_and_standardize_lz(data, verbose=False)
    shuff_and_standardize_ml(data, verbose=False)

    # Uniformity shuff and zscores.
    data["uniformity_zscore"] = stats.zscore(data["uniformity"])
    data["uniformity_shuff"] = \
        data["uniformity"].sample(frac=1).reset_index(drop=True)
    data["uniformity_shuff_zscore"] = stats.zscore(data["uniformity_shuff"])
        
    # Store adjusted DataFrame as csv.
    utils.store_language_data_to_csv(
        data, args.max_model_size, args.max_expr_len, 
        args.language_name, args.lang_gen_date, verbose=True
    )

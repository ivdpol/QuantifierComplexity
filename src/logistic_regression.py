""" 
This file is part of QuantifierComplexity.
"""
import argparse
import datetime
import os
from pathlib import Path
import dotenv   
import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
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
            "settings.",
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
        help="Date of language generation. Used to load the right csv file.",
    )
    parser.add_argument(
        "--csv_date",
        "-c",
        type=str,
        default=CSV_DATE,
        help="Date of csv file creation with the language data." + \
            "Used to load the right csv file.",
    )
    parser.add_argument(
        "--repeat",
        "-r",
        type=int,
        default=REPEAT,
        help="Number op samples to take for bootstrapping.",
    )
    parser.add_argument(
        "--sample_size",
        "-s",
        type=int,
        default=SAMPLE_SIZE,
        help="Size of samples used for bootstrapping.",
    )
    parser.add_argument(
        "--bootstrap_id",
        "-i",
        type=int,
        default=BOOTSTRAP_ID,
        help="Identifier of regression series.",
    )
    return parser.parse_args()


def bin_logistic_regression(
    data: pd.DataFrame, dependent_var: str, independent_vars: list, 
    print_summary=False
):
    ''' Performs a simple multiple regression, and prints the results.

    See https://www.statsmodels.org/devel/generated/statsmodels.
    discrete.discrete_model.LogitResults.html#statsmodels.discrete.
    discrete_model.LogitResults

    Args:
        data: A pandas dataframe
        dependent_var: A string. Name of a complexity measure
        independent_vars: A list of strings. Names of quantifier 
            properties.
        print_summary: True or False. Print the regression summary of
            each sample when True. Reports on convergence.

    '''
    formula = dependent_var + ' ~ ' + ' + '.join(independent_vars)
    logit_model = sm.logit(formula=formula, data=data)
    logit_estimates = logit_model.fit(disp=False, warn_convergence=True)
    if print_summary:
        print(logit_estimates.summary())
    return(logit_estimates)


def bootstrap_regression(
        csv_date: str, scores: list, dep_vars: list, regression_func, 
        repeat: int, sample_size: int, bootstrap_id: int, max_model_size: int, 
        max_expr_len: int, language_name: str, lang_gen_date: str,
        print_summary=False, verbose=False 
):
    '''Run regression on data sample and repeat.

    For each score (ind var) in scores, run a regression for the 
    dep_vars, and store coefficient results in a dataframe (per score).
    Do regression for orignal score data, randomly shuffled score data,
    and compute the difference between those coefficients.

    Args: 
        csv_date: A string. The date on which the csv data was created or
            last altered. For loading csv file with language data which  
            includes column names as given in dep_vars.
        scores: A list of strings. The names of the complexity measures:
            the independent variables.
        dep_vars: A list of strings. The names of the quantifier props:
            the dependent variables.
        regression_func: A function. Choice of regression function. 
        repeat: An int. The number of samples taken, i.e. the number
            regressions.
        sample_size: An int. The size of the samples taken.
        bootstrap_id: An int. Used for storing csv data with logistic
            regression data. Identifies the bootstrap series for a given
            date. Multiple regression sessions were done on the same data 
            to check for convergence.
        max_model_size: An int. Should coincide with the value in 
            max_model_size column in loaded csv data. Used for loading 
            csv data and storing regression data.
        max_expr_len: An int. Should coincide with the max value of
            expr_length column in loaded csv data. Used for loading csv data 
            and storing regression data.
        language_name: A string. Should coincide with the value of the 
            lot column in loaded csv data. Used for loading csv data and
            storing regression data.
        lang_gen_date: A string. The date on which the data was generated. 
            Used for loading csv data and storing regression data.
        print_summary: True or False. Print the regression summary of
            each sample when True. Reports on convergence.
        verbose: True or False. Print the regression results.

    '''
    data = utils.load_csv_data(
        max_model_size, max_expr_len, language_name, lang_gen_date, csv_date
    )
    results = {
        (score, dep_var): \
        pd.DataFrame() for score in scores for dep_var in dep_vars
    }
    # Take samples from original data set, do regression on
    # each sample and store parameter values of the 
    # regression results.
    for lap in range(repeat):
        if lap in np.arange(0, repeat + 1, repeat / 10):
            print(lap)
        for score in scores:
            # Reshuffle complexity scores.
            data[f"{score}_shuff_zscore"] = \
                data[
                    f"{score}_shuff_zscore"
                ].sample(frac=1).reset_index(drop=True)
        # Take sample.
        df_sample = data.sample(n=sample_size, replace=True)
        for score in scores:
            ind_vars = [
                f"{score}_zscore", # complexity (normalized)
                f"{score}_shuff_zscore" # complexity random baseline
            ]
            # Do regression on sample.
            for dep_var in dep_vars:
                for ind_var in ind_vars:
                    model = regression_func(
                        df_sample, dep_var, [ind_var], print_summary
                    )
                    # Store the coef of ind_var.
                    results[(score, dep_var)].at[lap, f"coef_{ind_var}"] = \
                        model.params[ind_var]
    for score in scores:
        ind_vars = [
            f"{score}_zscore", # complexity (normalized)
            f"{score}_shuff_zscore" # complexity random baseline
        ]
        for dep_var in dep_vars:
            # Store difference scores of coefficients: 
            # Original - Randomly shuffles.
            results[(score, dep_var)][f"{ind_vars[0]}-{ind_vars[1]}"] = \
                results[(score, dep_var)][f"coef_{ind_vars[0]}"] - \
                results[(score, dep_var)][f"coef_{ind_vars[1]}"]
            if verbose:
                print(f"results[({score}, {dep_var})]:")
                print(results[(score, dep_var)])
            # Store results.
            log_reg_date = datetime.datetime.now().strftime("%Y-%m-%d")
            
            csv_filename = utils.make_log_reg_csv_filename(
                ind_vars[0], dep_var, bootstrap_id, sample_size, repeat, 
                log_reg_date, max_model_size, max_expr_len, language_name
            )
            fileloc = utils.make_log_reg_csv_path(
                max_model_size, language_name, lang_gen_date, log_reg_date
            )
            results[(score, dep_var)].to_csv(
                fileloc / Path(csv_filename), index=False
            )


if __name__ == "__main__":
    # Default values for argparse args.
    LANGUAGE_NAME = "Logical_index" # "Logical_index"       # "Logical" 
    MAX_EXPR_LEN = 5                # 5 for Logical_index   # 7 for Logical
    MAX_MODEL_SIZE = 8 # 8
    LANG_GEN_DATE = "2022-03-11"    # "2022-03-11"  # "2020-12-25"
    CSV_DATE = "2022-03-16"         # "2022-03-16"  # "2021-05-05 for Logical_index
                                                    # "2021-03-23" for Logical
    SAMPLE_SIZE = 5000  # 5000
    REPEAT = 20000      # 20000
    BOOTSTRAP_ID = 1
    args = parse_args()
    
    # Set DataFrame print options.
    # #pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    # pd.set_option("display.max_colwidth", None)

    # Set input-parameters for bootstrap_regression function.
    REGRESSION_FUNC = bin_logistic_regression
    SCORES = ["lz_0", "lz_1", "lz_2", "lz_mean"]
    # ["ml"]
    # ["lz_0", "lz_1", "lz_2", "lz_mean"] these are the names for the
    # LZ scores based on different lexicographical permutations of the
    # quantifier representations (based on different lexicographical 
    # orderings of the models). 
    # score lz_mean stands for the mean value over lz_0, lz_1, and lz_2.
    # "ml" := minimal expression length
    # "lz" := Lempel Ziv complexity

    if "index" in args.language_name:
        quan_props = [
             "mon_quan_cons", "monotonicity", "quantity", "conservativity"
        ]
    else: 
        quan_props = ["mon_cons", "monotonicity", "conservativity"]
    QUAN_PROPS = quan_props
    
    bootstrap_regression(
        args.csv_date, SCORES, QUAN_PROPS, REGRESSION_FUNC,
        args.repeat, args.sample_size, args.bootstrap_id, args.max_model_size, 
        args.max_expr_len, args.language_name, args.lang_gen_date, 
        print_summary=False, verbose=False
    ) 
        
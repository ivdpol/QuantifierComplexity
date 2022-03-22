""" 
This file is part of QuantifierComplexity.
"""
import argparse
import datetime
import os
import sys
from pathlib import Path
import dotenv
import matplotlib.pyplot as plt    
import seaborn as sns 
import pandas as pd
from matplotlib.patches import Patch
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
        "--lang_gen_date",
        "-g",
        type=str,
        default=LANG_GEN_DATE,
        help="Date of language generation. Used to load the right csv file.",
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
    parser.add_argument(
        "--log_reg_date",
        "-d",
        type=str,
        default=LOG_REG_DATE,
        help="Date of csv file creation with the language data." + \
            "Used to load the right csv file.",
    )
    return parser.parse_args()


def distplot_log_reg_from_csv(
    ind_var1: str, ind_var2: str, dep_vars: list, sample_size: int, 
    repeat: int, log_reg_date: str, bootstrap_id: int, max_model_size: int, 
    max_expr_len: int, language_name: str, lang_gen_date: int
):
    '''Make and store distplots of existing logistic regression results.

    Plot coefficient distributions of regression over language data.
    For each property in dep_vars, make distplot of coefficient 
    distribution of raw score and randomly shuffled score.
    Below that, plot distribution of the difference in coefficient
    for raw scores and randomly shuffled scores.

    Args: 
        ind_var1: A string. Should be f"{score}zscore", i.e. zscore of
            raw score, for score in {"ml", "lz"}.
        ind_var1: A string. Should be f"{score}_shuff_zscore", i.e. 
            zscore of randomly shuffled score, for score in 
            {"ml", "lz"}.
        dep_vars: A list of strings. The names of the quantifier props:
            the dependent variables. For each there wil be a distplot.
        sample_size: An int. The size of the data samples that were used
            for the regressions.
        repeat: An int. The number of samples that were taken, 
            i.e. the number regressions.
        log_reg_date: A string. The date on which the regression data
            was made.
        bootstrap_id: An int. Used for loading csv data with logistic
            regression data. Identifies the bootstrap series for a given
            date. Multiple regression sessions were done on the same data 
            to check for convergence.
        max_model_size: An int. Used for loading and storing csv data.
            The maximum model size over which the meaning of quantifiers
            was computed in the language data.
        max_expr_len: An int. Used for loading and storing csv data.
            The maximum expression length of the quantifiers expressions
            in the language data.
        language_name: A string. Used for loading and storing csv data.
            Identifies the the collection of operators used for 
            generating the language data.
        lang_gen_date: A string. Used for loading and storing csv data.
            The date on which the language data was generated.
        print_summary: True or False. Print the regression summary of
            each sample when True. Reports on convergence.
        verbose: True or False. Print the regression results when True.

    '''
    dep_vars_names = {
        "mon_cons": "Both", "mon_quan_cons": "All", 
        "monotonicity": "Monotonicity", "quantity": "Quantity",
        "conservativity": "Conservativity"
    }
    csv_fileloc = utils.make_log_reg_csv_path(
        max_model_size, language_name, lang_gen_date, log_reg_date
    )
    if len(dep_vars) == 4:
        costum_figsize = (10, 5)
    if len(dep_vars) == 3:
        costum_figsize = (7.5, 5)
    fig, axs = plt.subplots(
        2, len(dep_vars), constrained_layout=True, figsize=costum_figsize
    )
    for ax in range(len(dep_vars)):
        colors = ["#e66101", "#fdb863", "#b2abd2", "#5e3c99"]
        # Colorbrew colors with diverging colorscheme:
        # colorblind safe, print friendly, photocopy safe.
        # From https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=4.
        # "#e66101" = dark orange, "#fdb863" = light orange 
        # "#b2abd2" = light purple, "#5e3c99" = dark purple
        csv_filename = utils.make_log_reg_csv_filename(
            ind_var1, dep_vars[ax], bootstrap_id, sample_size, repeat, 
            log_reg_date, max_model_size, max_expr_len, language_name
        )
        data = pd.read_csv(csv_fileloc / csv_filename)
        sns.distplot(
            data[f"coef_{ind_var1}"], color="#fdb863", ax=axs[0, ax], 
            label="Original"
        )
        sns.distplot(
            data[f"coef_{ind_var2}"], color="#e66101", ax=axs[0, ax], 
            label="Randomly shuffled"
        )
        sns.distplot(
            data[f"{ind_var1}-{ind_var2}"], ax=axs[1, ax], color="#5e3c99", 
            label="Original - Randomly shuffled"
        )
    for ax in axs.flat:
        ax.set_ylim(0, 15)
        ax.set_xlim(-1, 1)    
    axs[0, 0].set_ylabel("Density", fontsize=14)
    axs[1, 0].set_ylabel("Density", fontsize=14)

    for ax in range(1, len(dep_vars)):
        axs[0, ax].set_ylabel("")
        axs[1, ax].set_ylabel("")

    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    # plt.rcParams['legend.numpoints'] = 1
    # https://matplotlib.org/3.1.1/gallery/text_labels_and_
    # annotations/custom_legends.html
    custom_lines0 = [
        Patch(color="#fdb863", label="Original data"),
        Patch(color="#e66101", label="Randomly shuffled")
    ]
    custom_lines1 = [
        Patch(color="#5e3c99", label="Difference per sample")
    ]
    for ax in range(len(dep_vars)):
        axs[0, ax].set_xlabel("")
        axs[1, ax].set_xlabel("Coefficient", fontsize=14)
        axs[0, ax].set_title(dep_vars_names[dep_vars[ax]], size=16)
    for ax in range(1, len(dep_vars)):
        axs[0, ax].set_yticklabels([])
        axs[1, ax].set_yticklabels([])
    axs[0, ax].legend(handles=custom_lines0, loc="upper left", 
        bbox_to_anchor=(0.0, 1.0), fontsize=12
    )
    axs[1, ax].legend(handles=custom_lines1, loc="upper left", 
        bbox_to_anchor=(0.0, 1.0), fontsize=12
    )
    plot_fileloc = utils.make_log_reg_plot_path(
        max_model_size, language_name, lang_gen_date, log_reg_date
    )
    basic_filename = utils.make_basic_filename(
            max_model_size, max_expr_len, language_name  
    )
    plt.savefig(
        plot_fileloc / Path(f"{ind_var1}-{bootstrap_id}-repeat={repeat}-" + \
        f"samples={sample_size}-{basic_filename}.png"),
        dpi=600
    )
    plt.close()


def mean_and_CI_log_reg(
    ind_var1: str, ind_var2: str, dep_vars: list, sample_size: int, 
    repeat: int, log_reg_date: str, bootstrap_id: int, max_model_size: int, 
    max_expr_len: int, language_name: str, lang_gen_date: str, orig=False, 
    rand=False, diff=False, verbose=False
):
    '''Print mean and 95% CI of regression coefficients.

    The 95% CI of the real coefficient value, approximated by
    the 95 percentile of the bootstrap coefficient values:
    [percentile 0.025, percentile 0,975]

    Args: 
        ind_var1: A string. Should be f"{score}zscore", i.e. zscore of
            raw score, for score in {"ml", "lz"}.
        ind_var1: A string. Should be f"{score}_shuff_zscore", i.e. 
            zscore of randomly shuffled score, for score in 
            {"ml", "lz"}.
        dep_vars: A list of strings. The names of the quantifier props:
            the dependent variables. 
        sample_size: An int. The size of the data samples that were used
            for the regressions.
        repeat: An int. The number of samples that were taken, 
            i.e. the number regressions.
        log_reg_date: A string. The date on which the regression data
            was made.
        bootstrap_id: An int. Used for loading csv data with logistic
            regression results. Identifies the bootstrap series. Multiple 
            regression sessions were done on the same data to check for 
            convergence.
        max_model_size: An int. Used for loading and storing csv data.
            The maximum model size over which the meaning of quantifiers
            was computed in the language data.
        max_expr_len: An int. Used for loading and storing csv data.
            The maximum expression length of the quantifiers expressions
            in the language data.
        language_name: A string. Used for loading and storing csv data.
            Identifies the the collection of operators used for 
            generating the language data.
        lang_gen_date: A string. Used for loading and storing csv data.
            The date on which the language data was generated.
        orig: True or False. Print stats on regression coefficient for 
            original data when True.
        rand: True or False. Print stats on regression coefficient for 
            randomly shuffled data when True.
        diff: True or False. Print stats on difference between coefficient
             raw data and coefficient randomy shuffled data when True.
        verbose: True or False. Print the regression data when True.

    '''
    csv_fileloc = utils.make_log_reg_csv_path(
        max_model_size, language_name, lang_gen_date, log_reg_date
    )
    print("Mean, 95% CI")
    print("-" * 30)
    print(ind_var1[0:-7])
    for dep_var in dep_vars:
        print("-" * 30)
        print(dep_var)
        print("-" * 30)
        csv_filename = utils.make_log_reg_csv_filename(
            ind_var1, dep_var, bootstrap_id, sample_size, repeat, 
            log_reg_date, max_model_size, max_expr_len, language_name
        )
        data = pd.read_csv(csv_fileloc / csv_filename)
        if verbose:
            print("-" * 30)
            print("Logistic regression data")
            print("-" * 30)
            print(data)
            print("-" * 30)
            print(data.info())
        if orig:
            percentile_org = [
                round(data[f"coef_{ind_var1}"].quantile(0.025), 2), 
                round(data[f"coef_{ind_var1}"].quantile(0.975), 2)
            ]
            print(
                "Orig:", 
                round(data[f"coef_{ind_var1}"].mean(), 2),
                f"(95\\% CI {{{percentile_org}}})"
            )
        if rand:
            percentile_rand = [
                round(data[f"coef_{ind_var2}"].quantile(0.025), 2), 
                round(data[f"coef_{ind_var2}"].quantile(0.975), 2)
            ]
            print(
                "Rand:", 
                round(data[f"coef_{ind_var2}"].mean(), 2),
                f"(95\\% CI {{{percentile_rand}}})"
            )
        if diff:
            percentile_diff = [ 
                round(data[
                    f"{ind_var1}-{ind_var2}"
                ].quantile(0.025), 2), 
                round(data[
                    f"{ind_var1}-{ind_var2}"
                ].quantile(0.975), 2)
            ]
            print("Diff:", 
                round(data[f"{ind_var1}-{ind_var2}"].mean(), 2),
                f"(95\\% CI {{{percentile_diff}}})"
            )
    print("-" * 30)


def log_reg_plot_and_CI(
    scores: list, dep_vars: list, sample_size: int, repeat: int, 
    log_reg_date: str, bootstrap_id: int, max_model_size: int, 
    max_expr_len: int, language_name: str, lang_gen_date: str
):
    '''Make and store distplots and 95% CI of regression coefficients.

    Args: 
        scores: A list of strings. The names of the complexity measures:
            the independent variables. 
        dep_vars: A list of strings. The names of the quantifier props:
            the dependent variables. 
        sample_size: An int. The size of the data samples that were used
            for the regressions.
        repeat: An int. The number of samples that were taken, 
            i.e. the number regressions.
        log_reg_date: A string. The date on which the regression data
            was made.
        bootstrap_id: An int. Used for loading csv data with logistic
            regression data. Identifies the bootstrap series for a given
            date. Multiple regression sessions were done on the same data 
            to check for convergence.
        max_model_size: An int. Used for loading and storing csv data.
            The maximum model size over which the meaning of quantifiers
            was computed in the language data.
        max_expr_len: An int. Used for loading and storing csv data.
            The maximum expression length of the quantifiers expressions
            in the language data.
        language_name: A string. Used for loading and storing csv data.
            Identifies the the collection of operators used for 
            generating the language data.
        lang_gen_date: A string. Used for loading and storing csv data.
            The date on which the language data was generated.

    '''
    old_stdout = sys.stdout
    log_reg_plot_fileloc = utils.make_log_reg_plot_path(
        max_model_size, language_name, lang_gen_date, log_reg_date
    )
    scores_str = "_".join(scores)
    sys.stdout = utils.Logger(
            log_reg_plot_fileloc / 
            f"{scores_str}-mean_and_CI-{bootstrap_id}-{log_reg_date}.txt"
        ) 
    print("-" * 30)
    print("language \t\t", language_name)
    print("max_expr_len \t", max_expr_len)
    print("max_model_size \t", max_model_size)
    print("lang_gen_date \t", lang_gen_date)
    print("log_reg_date \t", log_reg_date)
    print("repeat \t\t\t", repeat)
    print("sample_size \t", sample_size)
    print("bootstrap_id \t", bootstrap_id)
    print("-" * 30)
    print()
    for score in scores: 
        ind_var1 = f"{score}_zscore" # complexity (normalized)
        ind_var2 = f"{score}_shuff_zscore" # complexity random baseline
        distplot_log_reg_from_csv(
            ind_var1, ind_var2, dep_vars, sample_size, repeat, log_reg_date, 
            bootstrap_id, max_model_size, max_expr_len, language_name, 
            lang_gen_date
        )
        mean_and_CI_log_reg(
            ind_var1, ind_var2, dep_vars, sample_size, repeat, log_reg_date, 
            bootstrap_id, max_model_size, max_expr_len, language_name, 
            lang_gen_date, 
            orig=False, rand=False, diff=True, verbose=False
        )
        print()
    sys.stdout = old_stdout 


if __name__ == "__main__":
    # Default values for argparse args.
    LANGUAGE_NAME = "Logical"     # "Logical_index"      # "Logical" 
    MAX_EXPR_LEN = 7              # 5 for Logical_index  # 7 for Logical
    MAX_MODEL_SIZE = 8     

    LANG_GEN_DATE = "2022-03-11"   
    # LANG_GEN_DATE = "2020-12-25" for SCORES = ["ml"]
    # LANG_GEN_DATE = "2022-03-11" 
        # for SCORES = ["lz_0", "lz_1", "lz_2", "lz_mean"]

    LOG_REG_DATE = "2022-03-16"     
    # LOG_REG_DATE = "2021-05-05" 
        # for SCORES = ["ml"] and LANGUAGE_NAME = "Logical_index"
    # LOG_REG_DATE = "2021-03-23" 
        # for SCORES = ["ml"] and LANGUAGE_NAME = "Logical"
    # LOG_REG_DATE = "2022-03-16" 
        # for SCORES = ["lz_0", "lz_1", "lz_2", "lz_mean"]

    SAMPLE_SIZE = 5000      #5000
    REPEAT = 20000          #20000
    BOOTSTRAP_ID = 1 
    # BOOTSTRAP_ID = 1 for SCORES = ["ml"]
    # BOOTSTRAP_ID = 3 for SCORES = ["lz_0", "lz_1", "lz_2", "lz_mean"]

    args = parse_args()

    if "index" in args.language_name:
        quan_props = [
             "mon_quan_cons", "monotonicity", "quantity", "conservativity"
        ]
    else: 
        quan_props = ["mon_cons", "monotonicity", "conservativity"]

    # Set input-parameters for making distplots.
    QUAN_PROPS = quan_props
    SCORES = ["lz_0", "lz_1", "lz_2", "lz_mean"]
    # ["ml"]
    # ["lz_0", "lz_1", "lz_2", "lz_mean"] these are the names for the
    # LZ scores based on different lexicographical permutations of the
    # quantifier representations (based on different lexicographical 
    # orderings of the models). 
    # score lz_mean stands for the mean value over lz_0, lz_1, and lz_2.
    

    log_reg_plot_and_CI(
        SCORES, QUAN_PROPS, args.sample_size, args.repeat, args.log_reg_date, 
        args.bootstrap_id, args.max_model_size, args.max_expr_len, 
        args.language_name, args.lang_gen_date
    )

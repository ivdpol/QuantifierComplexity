"""
This file is part of QuantifierComplexity.
"""
import os
import sys
import time
import datetime
import argparse
from pathlib import Path
import dotenv
import utils
import languagegenerator as lg


# Load environment variables from .env (which is in same dir as src).
# Don't forget to set "PROJECT_DIR" in .env to the name of the location 
# from which you are running current source code.
dotenv.load_dotenv(dotenv.find_dotenv())
# Set paths to relevant directories. 
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
RESULTS_DIR = PROJECT_DIR / RESULTS_DIR_RELATIVE
utils.make_directory_if_not_present(RESULTS_DIR)
 

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
        help="Models up to this size will be used to evaluate the meaning" + \
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
        "--store_at_each_length",
        "-p",
        type=int,
        default=1,
        help="If 1, will store the generated expressions at each round" + \
            "of lengths",
    )
    return parser.parse_args()


if __name__ == "__main__": 
    start = time.time()
    date_obj = datetime.datetime.now()
    date = date_obj.strftime("%Y-%m-%d")
    date_and_time = date_obj.strftime("%Y-%m-%d %H:%M:%S")
    clock = date_obj.strftime("%H.%M.%S")
    REDIRECT_STDOUT = True
    LANGUAGE_NAME = "Logical_index"
    MAX_EXPR_LEN = 5
    MAX_MODEL_SIZE = 8
    args = parse_args()

    if REDIRECT_STDOUT:
        # Make relevant directories.
        language_dir = utils.make_language_dir(
            args.max_model_size, args.language_name
        )

        file_loc = utils.make_stdout_dir(
            args.max_model_size, args.max_expr_len, args.language_name, date
        )
        sys.stdout = utils.Logger(
            file_loc / f"{args.max_expr_len}=max-expr-len_time={clock}.txt"
        )

    print('\n************* JOB *************')
    print('max_model_size =', args.max_model_size)
    print('max_expr_len =', args.max_expr_len)
    print('language_name =', args.language_name)
    print('date and time =', date_obj.strftime("%Y-%m-%d %H:%M:%S")
    )
    print('*******************************\n')

    # Make language generator object.
    l = lg.LanguageGenerator( 
        args.max_model_size, args.dest_dir, args.language_name, 
        args.store_at_each_length
    )
    exps = l.gen_all_expr_and_their_scores(args.max_expr_len)
    finish = time.time()
    print("running time =", (finish-start) / 60, "\n")

    l.test_prefilter()

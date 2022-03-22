'''
This file is part of QuantifierComplexity.
'''
import math
import sys
import contextlib
import os
import re
import datetime
import itertools as it
from pathlib import Path
from collections import defaultdict, Counter
import dill
import numpy as np
import pandas as pd
import dotenv


# Load environment variables from .env (which is in same dir as src).
# Don't forget to set "PROJECT_DIR" in .env to the name of the location 
# from which you are running current source code.
dotenv.load_dotenv(dotenv.find_dotenv())
# Set paths to relevant directories.
# Set directory names.
PROJECT_DIR = Path(os.getenv("PROJECT_DIR"))
RESULTS_DIR_RELATIVE = os.getenv("RESULTS_DIR_RELATIVE")
RESULTS_DIR = PROJECT_DIR / RESULTS_DIR_RELATIVE

def make_distplot_filename(
    ind_var: str, bootstrap_id: int, sample_size: int, repeat: int, 
    plot_date: str, max_model_size: int, max_expr_len: int, language_name: str
):
    basic_filename = make_basic_filename(
        max_model_size, max_expr_len, language_name
    )
    return f"{ind_var}-{bootstrap_id}-repeat={repeat}-samples=" + \
        f"{sample_size}-{basic_filename}-{plot_date}.png"


def make_log_reg_csv_filename(
    ind_var: str, dep_var: str, bootstrap_id: int, sample_size: int, 
    repeat: int, log_reg_date: str, max_model_size: int, max_expr_len: int, 
    language_name: str
):
    basic_filename = make_basic_filename(
        max_model_size, max_expr_len, language_name
    )
    return f"{ind_var}-{bootstrap_id}-repeat={repeat}-samples=" + \
        f"{sample_size}-{basic_filename}-{dep_var}-{log_reg_date}.csv"
 

def dataframes_to_txt(dataframes: list, file_loc: str, filename: str):
    file = open(f"{file_loc / filename}.txt", "w")
    for df in dataframes:
        file.write(df.to_string() + "\n\n")
    file.close()


def make_stdout_dir(
    max_model_size: int, max_expr_len: int, language_name: str, 
    lang_gen_date: str
):
    language_dir = make_language_dir(max_model_size, language_name)
    date_dir = make_date_dir(language_dir)
    stdout_dir = date_dir / "stdout" 
    make_directory_if_not_present(stdout_dir)
    return stdout_dir


def make_log_reg_csv_path(
    max_model_size: int, language_name: str, lang_gen_date: str,
    log_reg_date: str
):
    '''Folder withing analysis repository.'''
    if not log_reg_date:
        log_reg_date = datetime.datetime.now().strftime("%Y-%m-%d")
    analysis_path = make_analysis_path(
        max_model_size, language_name, lang_gen_date
    )
    log_reg_path = analysis_path / "log_regression" 
    make_directory_if_not_present(log_reg_path)
    make_directory_if_not_present(log_reg_path / log_reg_date)
    make_directory_if_not_present(log_reg_path / log_reg_date / "csv")
    return log_reg_path / log_reg_date / "csv"


def make_log_reg_plot_path(
    max_model_size: int, language_name: str, lang_gen_date: str,
    log_reg_date: str
):
    '''Folder within analysis repository.'''
    analysis_path = make_analysis_path(
        max_model_size, language_name, lang_gen_date
    )
    log_reg_path = analysis_path / "log_regression" 
    plot_date = datetime.datetime.now().strftime("%Y-%m-%d")
    make_directory_if_not_present(log_reg_path)
    make_directory_if_not_present(log_reg_path / log_reg_date)
    make_directory_if_not_present(log_reg_path / log_reg_date / "plots")
    make_directory_if_not_present(
        log_reg_path / log_reg_date / "plots" / plot_date
    )
    return log_reg_path / log_reg_date / "plots" / plot_date


def make_descriptive_stats_path(
    max_model_size: int, language_name: str, lang_gen_date: str,
    descr_stats_date=None
):
    '''Folder withing analysis repository.'''
    if not descr_stats_date:
        descr_stats_date = datetime.datetime.now().strftime("%Y-%m-%d")
    analysis_path = make_analysis_path(
        max_model_size, language_name, lang_gen_date
    )
    descr_stats_path = analysis_path / "descriptive_stats" 
    make_directory_if_not_present(descr_stats_path)
    make_directory_if_not_present(descr_stats_path / descr_stats_date)
    return descr_stats_path / descr_stats_date


def make_analysis_path(
    max_model_size: int, language_name: str, lang_gen_date: str
):
    language_dir_name = make_language_dir_name(max_model_size, language_name)
    analysis_path = Path(RESULTS_DIR / language_dir_name / lang_gen_date 
        / "analysis"
    )
    make_directory_if_not_present(analysis_path)
    return analysis_path


def make_test_path(
    max_model_size: int, max_expr_len: int, language_name: str, 
    lang_gen_date: str, test_date: str
):
    language_dir_name = make_language_dir_name(max_model_size, language_name)
    test_path = Path(RESULTS_DIR / language_dir_name / lang_gen_date 
        / "tests" / test_date
    )
    make_directory_if_not_present(test_path)
    return test_path


def make_test_dir(max_model_size: int, language_name: str):
    '''Make and return directory "test" in given repository.

    Args:
        
        max_model_size: An int. Max size of models on which expressions 
            were evaluated.
        language_name: A string. Name of the json file with the  
            corresponding language settings. Defines the collection 
            of operators by which the language was generated.

    Returns: Path object to language directory.
    
    '''
    language_dir = make_language_dir(max_model_size, language_name)
    language_date_dir = make_date_dir(language_dir)
    test_dir = language_date_dir / "tests"
    make_directory_if_not_present(test_dir)
    return test_dir


def make_language_dir(max_model_size: int, language_name: str):
    '''Make and return directory with language name and max_model_size.

    Args:
        max_model_size: Integer.
        language_name: Refers to language setting that can be found in
            language_name.json file.

    Returns: Path object to language directory.
    
    '''
    language_dir_name = make_language_dir_name(
        max_model_size, language_name
    )
    language_dir = Path(RESULTS_DIR) / language_dir_name
    make_directory_if_not_present(language_dir)
    return language_dir


def make_date_dir(directory, date=None):
    '''Make and return date directory in given directory path.

    Args:
        directory: Path to existing directory, in which a repository
            with the name of date will be made.
        date: A string representing a date, in the form "%Y-%m-%d".

    '''
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
    make_directory_if_not_present(directory / date)
    return directory / date


def make_language_dir_name(max_model_size: int, language_name: str):
    '''Return name of directory for results of given settings.

    Only depends on max_model_size, and language_name. 

    '''
    if language_name is not None:
        return f"Language={language_name}-max_model_size={max_model_size}"
    return f"max_model_size={max_model_size}"


def make_test_file_name(
    max_model_size: int, max_expr_len: int, language_name: str
):
    '''Return name of test_file for results of given settings. 

    Only depends on model_size, max_expr_len, and language_name.

    '''
    return f"test-data-language={language_name}-max_model_size=" + \
        f"{max_model_size}-max-expr-len={max_expr_len}"


def make_csv_filename(
    max_model_size: int, max_expr_len: int, language_name: str
):
    '''Return name of csv_file for results of given settings. 

    Only depends on model_size, max_expr_len, and language_name.

    '''
    return f"language={language_name}-max_model_size={max_model_size}" + \
        f"-max-expr-len={max_expr_len}.csv"


def make_basic_filename(
    max_model_size: int, max_expr_len: int, language_name: str
):
    '''Return filename for results of given settings. 

    Only depends on model_size, max_expr_len, and language_name.

    '''
    return f"language={language_name}-max_model_size={max_model_size}" + \
        f"-max-expr-len={max_expr_len}"


def make_csv_path(
    max_model_size: int, language_name: str, lang_gen_date: str, csv_date: str
):
    language_dir_name = make_language_dir_name(max_model_size, language_name)
    csv_path = Path(
        RESULTS_DIR / language_dir_name / lang_gen_date / "csv" / csv_date 
    )
    make_directory_if_not_present(csv_path)
    return csv_path


def load_csv_data(
    max_model_size: int, max_expr_len: int, language_name: str, 
    lang_gen_date: str, csv_date: str
):
    '''Load csv data into Pandas DataFrame.

    Args:
        lang_gen_date: String of the form "%Y-%m-%d". Creation date of  
            language generator.
        csv_date: String of the form "%Y-%m-%d". Creation date of 
            csv_file.

    '''
    csv_path = make_csv_path(
        max_model_size, language_name, lang_gen_date, csv_date
    )
    csv_filename = make_csv_filename(
        max_model_size, max_expr_len, language_name
    )
    return pd.read_csv(csv_path / csv_filename)


def store_language_data_to_csv(
    data: pd.DataFrame, max_model_size: int, max_expr_len: int, 
    language_name: str, lang_gen_date: str, verbose=False
):
    '''Load csv data into Pandas DataFrame.

    Args:
        lang_gen_date: String of the form "%Y-%m-%d". Creation date of  
            language generator.

    '''
    csv_date = datetime.datetime.now().strftime("%Y-%m-%d")
    csv_path = make_csv_path(
        max_model_size, language_name, lang_gen_date, csv_date
    )
    csv_filename = make_csv_filename(
        max_model_size, max_expr_len, language_name
    )
    if verbose:
        print("-"*30 + " stored data " + "-"*30)
        print(data)
        print("-"*30 + " end data " + "-"*30)
    data.to_csv(csv_path / csv_filename, index=False) 


def get_exp_len(expr: tuple) -> int:
    '''Compute the length of an expression.

    The length of an expression is defined by the number of operators 
    it contains. Shortcut to compute this: count the number of left 
    brackets in a tuple, by making it into a string and going through 
    its characters.

    Args:
        expr: Represents a quantifier expression created by a 
        languagegenerator object. 

    Returns:
        The number of left brackets in expr: the length of the 
        expression.

    '''
    nr_brackets = len([symbol for symbol in str(expr) if symbol == '('])
    return str(nr_brackets)
    

def get_key(val, my_dict):
    return list(my_dict.keys())[list(my_dict.values()).index(val)] 


def is_unique(data: pd.DataFrame):
    '''Check wheter all entries in data are the same.'''
    array = data.to_numpy() # data.values (pandas<0.24)
    return (array[0] == array).all()
 

def make_directory_if_not_present(directory):
    '''Make directory if it does not exist yet'''
    if not os.path.exists(directory):
        os.mkdir(directory)


def expr_len(expr: tuple):
    '''Returns expression length, given expresion in tuple format. 

    Expression length is defined as the number of operators in an 
    expression.

    '''
    # Base case: when none of the elements of the expr is a tuple.
    if tuple not in {type(elem) for elem in expr}:
        return 1
    # Recursive case.
    return sum(
        expr_len(elem) for elem in expr[1:] if isinstance(elem, tuple)
    ) + 1 


def prettify_model(model):
    '''Prettifies model representation.'''
    tups = list(zip(*model))
    out = []
    for a in tups:
        if a[0] and a[1]:
            out.append("AandB")
        elif a[0] and not a[1]:
            out.append("A")
        elif not a[0] and a[1]:
            out.append("B")
        elif not a[0] and not a[1]:
            out.append("M")
    return out


def same_exp(exp1, exp2):
    '''Checks if 2 expressions are syntactically the same.'''
    perm_invariant_ops = ["union", "=f", "=", "+", "intersection", "and", "or"]
    if exp1 == exp2:
        return True
    if (type(exp1) != type(exp1)) or (len(exp1) != len(exp2)):
        return False
    if type(exp1) == type(exp2) == tuple:
        if len(exp1) == 3:
            if exp1[0] == exp2[0]:
                if exp1[0] in perm_invariant_ops:
                    return (
                        same_exp(exp1[1], exp2[1])
                        and same_exp(exp1[2], exp2[2])
                    ) or (
                        same_exp(exp1[1], exp2[2])
                        and same_exp(exp1[2], exp2[1])
                    )
                return same_exp(exp1[1], exp2[1]) and same_exp(
                    exp1[2], exp2[2]
                )
        return exp1[0] == exp2[0] and same_exp(exp1[1], exp2[1])
    if isinstance(exp1, str):
        return exp1 == exp2
    return False


def return_if_unique(lang_generator, expr):
    meaning = lang_generator.compute_meaning(expr)
    output_type = lang_generator.get_output_type(expr)
    if any(
        lang_generator.same_meaning(meaning, m)
        for m in lang_generator.output_type2expression2meaning[
            output_type
        ].values()
    ):
        pass
    else:  # unique meaning
        return (output_type, expr, meaning)


def return_meaning_matrix(exps, lang_gen):
    '''Return the extensions (bin reps of meanings) of expressions.

    Given a sorted list with expressions and a language generator, 
    returns matrix where the ith row corresponds to the meaning of 
    the ith expression.

    '''
    nof_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    out = np.zeros((len(exps), nof_models))
    for i, exp in enumerate(exps):
        out[i, :] = lang_gen.output_type2expression2meaning[bool][exp]
    return out


def divide_possibly_zero(array_a, array_b):
    '''return 0 for nans'''
    out = array_a / array_b
    out[np.isnan(out)] = 0
    return out


def element_wise_binary_entropy(prob_df):
    if isinstance(prob_df, pd.DataFrame):
        # Ignore the cases of dividing by zero. This only happens
        # for the 0 *log2(0) case, which gives 0 * -inf = NaN, and 
        # that is set to 0 afterwards. This is according to the 
        # definition of binary entropy: taking log2(0) to be 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            return pd.DataFrame(
                data=(
                    - prob_df.values * np.log2(prob_df.values)
                    - (1 - prob_df.values) * np.log2(1 - prob_df.values)
                ),
                index=prob_df.index,
                columns=prob_df.columns,
            ).fillna(0)
    else:
        with np.errstate(divide='ignore', invalid='ignore'):
            return pd.Series(
                data=(
                    - prob_df.values * np.log2(prob_df.values)
                    - (1 - prob_df.values) * np.log2(1 - prob_df.values)
                ),
                index=prob_df.index,
            ).fillna(0)


def h(q):
    '''Binary entropy func'''
    if q in {0, 1}:
        return 0
    return (q * math.log(1 / q, 2)) + ((1 - q) * math.log(1 / (1 - q), 2))


def return_char_array(m):
    out = [0, 0, 0, 0]
    for a, b in zip(*m):
        if a and b:
            out[0] += 1
        elif a:
            out[1] += 1
        elif b:
            out[2] += 1
    return np.array(out)
 

def compute_char_tuple_matrix(lang_gen):
    '''Retrun characteristic tuple.

    Given lang generator, returns 4 x m matrix, m begin the number of 
    models, and each vector representing the characterizing tuple.

    Args: 
        lang_gen: A language generator object.

    Returns: A np.array

    '''
    n_of_models = sum(
        lang_gen.number_of_subsets ** i
        for i in range(1, lang_gen.max_model_size + 1)
    )
    out = np.zeros((n_of_models, 4), dtype=np.uint8)
    for i, m in enumerate(lang_gen.generate_universe()):
        out[i, :] = return_char_array(m)
    return out


def init_char_tuple_distribution(char_matrix):
    '''Return characteristic tuple counter.

    Returns Counter that maps char tuple to their probability, and 
    another dict that maps char tuple to a bool idx array that can 
    be used to select the relevant models.

    '''
    char_tuple2count = Counter()
    char_tuple2idxs = defaultdict(list)
    for i, char_array in enumerate(char_matrix):
        tuple_repr = tuple(char_array)
        char_tuple2count[tuple_repr] += 1
        char_tuple2idxs[tuple_repr].append(i)
    n_of_models = char_matrix.shape[0]
    char_tuple2freq = Counter()
    for char_tuple, count in char_tuple2count.items():
        char_tuple2freq[char_tuple] = count / n_of_models
    char_tuple2bool_idxs = dict()
    for char_tuple, idxs in char_tuple2idxs.items():
        char_tuple2bool_idxs[char_tuple] = np.array(
            [i in idxs for i in range(n_of_models)]
        )
    return (char_tuple2freq, char_tuple2bool_idxs)


def load_expressions_for(
    language_name: str, max_model_size: int, max_expression_length=None
):
    '''Load expression up to (and including) max expr len found.

    Given name of language and max model size, loads expressions up to
    greatest length possible (i.e. that have been generated so far),
    or up to max expression length if given.

    Args:
        language_name: A string. Name of the json file with the  
            corresponding language settings. Defines the collection 
            of operators by which the language was generated.
        max_model_size: An int. Max size of models on which expressions 
            were evaluated.
        max_expression_lenth: Max length of the expressions in the
            language
    
    Returns: List with all expressions in tuple format.

    '''
    language_dir = (
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / make_language_dir_name(max_model_size, language_name)
    )
    if max_expression_length is None:
        to_load = max(
            [f for f in os.listdir(language_dir) if "expressions" in f],
            key=lambda x: re.findall("[0-9]+", x)[-1],
        )
    else:
        for f in os.listdir(language_dir):
            if "expressions" not in f:
                continue
            if re.findall("[0-9]+", f):
                n = re.findall("[0-9]+", f)[-1]
                if n == str(max_expression_length):
                    to_load = f
                    break
        else:
            print("Could not find file with given settings")
    with open(Path(language_dir) / to_load, "rb") as f:
        out = dill.load(f)
    return out


def load_lang_gen_for(
    language_name: str, max_model_size: int, lang_gen_date: str,
    max_expression_length=None
):
    '''Load language generator.

    Given name of language and max model size, load language generator 
    for highest possible number of expressions from results dir of that
    language.

    Args:
        language_name: A string. Name of the json file with the  
            corresponding language settings. Defines the collection 
            of operators by which the language was generated.
        max_model_size: An int. Max size of models on which expressions 
            were evaluated.
        max_expression_lenth: Max length of the expressions in the
            language. 

    '''
    lang_gen_dir = (
        Path(PROJECT_DIR)
        / Path(RESULTS_DIR_RELATIVE)
        / make_language_dir_name(max_model_size, language_name)
        / lang_gen_date / "language_generators"
    )
    if max_expression_length is None:
        to_load = max(
            [f for f in os.listdir(lang_gen_dir) if "lang" in f],
            key=lambda x: re.findall("[0-9]+", x)[-1],
        )
    else:
        for f in os.listdir(lang_gen_dir):
            if "lang" not in f:
                continue
            if re.findall("[0-9]+", f):
                n = re.findall("[0-9]+", f)[-1]
                if n == str(max_expression_length):
                    to_load = f
                    break
        else:
            print("Could not find file with given settings")
        # print(os.listdir(lang_gen_dir))
        # print(
        #     [[-1] for f in os.listdir(lang_gen_dir)]
        # )
        # to_load = next(
        #     f
        #     for f in os.listdir(lang_gen_dir)
        #     if "lang" in f
        #     and re.findall("[0-9]+", f)[-1] == str(max_expression_length)
        # )
    with open(Path(lang_gen_dir) / to_load, "rb") as f:
        out = dill.load(f)
    return out


def tuple_format(model):
    '''Convert a model to it's tuple format'''
    return tuple((int(a), int(b)) for a, b in zip(*model))


class Logger():
    '''Tool to simultaneously save stdout to file and print it.'''
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.stdout.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.stdout.flush()
        self.log.flush()

# Used for not printing stdout.
class DummyFile(object):
    def write(self, x): pass


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout


def pretty_print_expr(expr: tuple):
    '''Return human-readable representation of an expression.'''

    if str(expr).isdigit() or expr in ["A", "B"]:
        return str(expr)

    if expr == (">", "0", "0"):
        return "\u22a5"
    elif expr == (">", "1", "0"):
        return "\u22a4"

    main_operator = expr[0]
    if main_operator == "union":
        return "({} \u222a {})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "intersection":
        return "({} \u2229 {})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator in [">", "<", "="]:
        return "({} {} {})".format(
            pretty_print_expr(expr[1]),
            main_operator,
            pretty_print_expr(expr[2])
        )
    elif main_operator == "subset":
        return "{} \u2286 {}".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "diff":
        return "({} \\ {})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "and":
        return "({} \u2227 {})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "or":
        return "({} \u2228 {})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "not":
        return "(\u00ac {})".format(
            pretty_print_expr(expr[1])
        )
    elif main_operator == "index":
        return "i({},{})".format(
            pretty_print_expr(expr[1]),
            pretty_print_expr(expr[2])
        )
    elif main_operator == "card":
        return "|{}|".format(
            pretty_print_expr(expr[1])
        )
    else:
        return str(expr)

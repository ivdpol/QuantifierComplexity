"""
This file is part of QuantifierComplexity.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import math
import datetime
import argparse
from pathlib import Path
import dill
import dotenv
import utils
import quantifier_properties
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
        help="Name of json file (when adding .json) that specifies " + \
            "settings",
    )
    parser.add_argument(
        "--lang_gen_date",
        "-g",
        type=str,
        default=LANG_GEN_DATE,
        help="Date of language generation. Used to load the right csv file. ",
    )
    return parser.parse_args()


def convert_decimal_to_ternary(N=int):
	'''Return given decimal number as ternary number in type int.

    Args:
        N: An int. Represents a decimal number.

    Returns: 
    	A string. Represents a ternary number that is equal in value
    	to the decimal number N.

    '''
	return np.base_repr(N, base=3)


def convert_ternary_to_decimal(N=int):
	'''Return given ternary number as decimal number in type int.

	Code by Code_Mech, found on 
	www.geeksforgeeks.org/ternary-number-system-or-base-3-numbers/

    Args:
        N: An int. Represents a ternary number.

    Returns: 
    	An int. Represents a decimal number that is equal in value
    	to the ternary number N.

    '''
	# If the decimal number is greater than 0,  
	# compute the decimal representation of the number.
	if (N != 0):
		decimalNumber = 0
		i = 0
		remainder = 0
		# Loop to iterate through the number.
		while (N != 0):
			remainder = N % 10
			N = N // 10
			# Computing the decimal digit.
			decimalNumber += int(remainder * math.pow(3, i))
			i += 1
		return decimalNumber
	else:
		return 0


def lex_base_order_perm(seq_1=list, seq_2=list, N=int):
	'''Return the per digit permutation from seq_1 to seq_2.

	Seq_1 and seq_2 represent a lexicographical base order, 
	used to enumeratie a universe of models in a lexicographical order 
	(by first encoding and then ordering the models based on this 
	lex base order).

	Args: 
		seq_1: A list. Should be of equal length and contain the same
			symbols as seq_2. Sould contain exactly the integers 
			0, ..., len(seq_1) - 1 as value entries.
		seq_2: A list. Should be of equal length and contain the same
			symbols as seq_1. Sould contain exactly the integers 
			0, ..., len(seq_1) - 1 as value entries.
		N: An int. Should be a value entry of both seq_1 and seq_2.
	
	Returns: An int. Given the index of value entry N in seq_1,
		returns the value entry at that index in seq_2.

	'''

	if len(seq_1) != len(seq_2):
		raise ValueError("Supply seq_1 and seq_2 of equal length")
	sort_seq_1 = sorted(seq_1)
	sort_seq_2 = sorted(seq_2)

	if sorted(seq_1) != sorted(seq_2):
		raise ValueError("Supply seq_1 and seq_2 with identical " + 
			"value entries.")
	# TODO: posibly raise more ValueError's to check whether the 
	# inputs are correct.

	idx = seq_1.index(N)
	return seq_2[idx]


def lex_perm(
	quan_meaning=list, base_order=list, perm_order=list, model_size=int
):
	'''Return permuted quan_meaning.

	Works for models with 3 subareas ("AnotB", "AandB", "BnotA").
	A quantifier meaning representation (for a given model size) 
	is a binary sequence that corresponds to a model sequence. 
	The model sequence is made as follows. Each model area gets
	assigned a symbol: 0, 1, 2. The ordered objects in the model are
	then labeld according to the area symbol (depending on the area
	that the object is in). A model of size 3, with all objects in 
	area 0, is encoded by 000. Then the models are placed in a 
	sequence, in lexicographical order over a base order, e.g., 
	over base_order (0,1,2). Then if a quantifier is true in 
	the i-th model of that sequence, a 1 is put at the i-th place
	of the quantifier meaning representation.
	When the models sequence is done in lexicographical order
	over a permutation of the original bse order (e.g., over
	perm_order (0,2,1)), then this results in a permutation
	of the quantifier meaning. This function computes such a
	permutation.
	

	Args:
		quan_meaning: A list with only 1's and 0's. Represents a 
			quantifier extension in lexicographical order over the 
			quantifier models, based on base_order. (It actually
			works for a list with any kind of value entries.
			Also works for np.array instead of list.)
		base_order: A list with entries 0,1,2 in some order.
		perm_order: A list with entries 0,1,2 in some order.
		model_size: An int. Represents the size of the models over 
			which the meaning of the quantifier is represented. 

	Returns: A list. A permutation of quan_meaning. Represents the 
		quantifier meaning in the lexicographical order over 
		perm_order.

	'''

	# Make dummy list to iteratively fill the meaning permutation.
	perm_quan_meaning = ["x" for _ in range(len(quan_meaning))]
	# Fill each entry of perm_quan_meaning dummy list.
	for idx in range(len(quan_meaning)):
		# Write decimal idx as ternary number in type list 
		# of lengt model_size. 
		ternary_idx = [int(digit) for digit in convert_decimal_to_ternary(idx)]
		# If needed, add zero's to the left.
		ternary_idx = [
			0 for _ in range(model_size - len(ternary_idx))
		] + ternary_idx
		# Make permutation of ternary_idx, by per digit permutation.
		# Make dummy list to iteratively fill.
		perm_ternary_idx = ["x"  for _ in range(model_size)]
		for pos in range(model_size):
			perm_ternary_idx[pos] = lex_base_order_perm(
			 	base_order, perm_order, ternary_idx[pos]
			)
		# Get permutation of decimal idx position.
		# 
		perm_ternary_idx_int = int("".join(map(str, perm_ternary_idx)))
		perm_idx = convert_ternary_to_decimal(perm_ternary_idx_int)
		# Fill the value entry at original decimal idx position 
		# of quan_meaning at its permuted idx position 
		# in perm_quan_meaning.
		perm_quan_meaning[idx] = quan_meaning[perm_idx]
	return perm_quan_meaning


def lex_perm_max_model_size(
	quan_meaning=list, base_order=list, perm_order=list, max_model_size=int
):
	'''
	Args:
		quan_meaning: A list with only 1's and 0's. Represents a 
			quantifier extension in lexicographical order over the 
			quantifier models, based on base_order. (It actually
			works for a list with any kind of value entries.
			Also works for np.array instead of list.)
		base_order: A list with entries 0,1,2 in some order.
		perm_order: A list with entries 0,1,2 in some order.
		max_model_size: An int. Represents the max size of the models  
			over which the meaning of the quantifier is represented. 

	Returns: A list. A permutation of quan_meaning. Represents the 
		quantifier meaning in the lexicographical order over 
		perm_order per model size, from small to large model_size.
	
	'''	
	# Make dummy list to fill the meaning permutation,
	# per model size block.
	perm_quan_meaning = ["x" for _ in range(len(quan_meaning))]
	# Per model size, take the portion of quan_meaning that corresponds
	# with models of that size, and permute it from base_order
	# to perm_order.
	start_of_block = 0
	for model_size in range(1, max_model_size + 1):
		end_of_block = start_of_block + 3**model_size 
		perm_quan_meaning_block = lex_perm(
			quan_meaning[start_of_block:end_of_block], 
			base_order, perm_order, model_size
		)
		# Fill the dummy list with the permutation for current block.
		perm_quan_meaning[
			start_of_block:end_of_block
		] = perm_quan_meaning_block
		start_of_block = end_of_block
	return perm_quan_meaning

	
if __name__ == "__main__": 
    
    start = time.time()
    date_obj = datetime.datetime.now()
    date = date_obj.strftime("%Y-%m-%d")
    clock = date_obj.strftime("%H.%M.%S")
    date_and_time = date_obj.strftime("%Y-%m-%d %H:%M:%S")
    print("\nSTART =", date_and_time)
    
    LANGUAGE_NAME = "Logical" 		# "Logical_index"       # "Logical" 
    MAX_EXPR_LEN = 7                # 5 for Logical_index   # 7 for Logical
    MAX_MODEL_SIZE = 8
    LANG_GEN_DATE = "2022-03-12"    #"2022-03-12"  			#"2022-03-12"     
    args = parse_args()

    REDIRECT_STDOUT = True

    if REDIRECT_STDOUT:
        # Make relevant directories if not yet there.
        language_dir_name = utils.make_language_dir_name(
            args.max_model_size, args.language_name
        )
        csv_path = Path(
            RESULTS_DIR / language_dir_name / args.lang_gen_date / "csv" 
        )
        stdout_dir = csv_path / "stdout" 
        utils.make_directory_if_not_present(csv_path)
        utils.make_directory_if_not_present(stdout_dir)
        sys.stdout = utils.Logger(
            stdout_dir / f"{args.max_expr_len}=max-expr-len_time={clock}.txt"
        )

    print("\n************* JOB *************")
    print("max_model_size =", args.max_model_size)
    print("max_expr_len =", args.max_expr_len)
    print("language_name =", args.language_name)
    print("date and time =", date_obj.strftime("%Y-%m-%d %H:%M:%S")
    )
    print("activity = compute lempel-ziv complexity over lexical permutations")
    print("*******************************\n")

    # Set DataFrame print options.
    # pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    # pd.set_option("display.width", None)
    # pd.set_option("display.max_colwidth", None)

    # Load language generator object.
    lang_gen = utils.load_lang_gen_for(
    	args.language_name, args.max_model_size, args.lang_gen_date,
    	max_expression_length=args.max_expr_len
    )
    print("\nBIG DATA TABLE a.k.a. ORIGINAL DATA\n", lang_gen.big_data_table)

    # Get expressions and extensions from original data.
    meaning_matrix = lang_gen.get_meaning_matrix()
    # print(meaning_matrix)
    expr2prop0 = pd.Series(
    	index=lang_gen.big_data_table["expression"], dtype=float
    )
    expr2prop1 = pd.Series(
    	index=lang_gen.big_data_table["expression"], dtype=float
    )
    expr2prop2 = pd.Series(
    	index=lang_gen.big_data_table["expression"], dtype=float
    )
    exprs = lang_gen.type2expr2meaning[bool]
    print("\nMEANING MATRIX EXPRESSIONS\n", expr2prop0)

    # Compute Lempel-Ziv complexity over permuted meaning extensions.
    quan_prop = quantifier_properties.LempelZiv
    for expr, meaning in exprs.items():
    	expr2prop0[expr] = quan_prop.property_function(meaning)
    	perm1_meaning = np.array(
    		lex_perm_max_model_size(meaning, [0, 1, 2], [1, 2, 0], 
    		args.max_model_size)
    	)
    	expr2prop1[expr] = quan_prop.property_function(perm1_meaning)
    	perm2_meaning = np.array(
    		lex_perm_max_model_size(meaning, [0, 1, 2], [2, 0, 1], 
    		args.max_model_size)
    	)
    	expr2prop2[expr] = quan_prop.property_function(perm2_meaning)

    # Combine series with Lempel-Ziv complexity over permuted meaning 
    # extensions into one dataframe.
    perm_data = pd.concat(
    	[expr2prop0, expr2prop1, expr2prop2], axis=1
    ).reset_index()
    perm_data.columns = [
        "expression_perm", "lempel_ziv_0", "lempel_ziv_1", "lempel_ziv_2"
    ]

    # Add mean value of Lempel-Ziv complexities over the three 
    # permuted meaning extensions (over the three lexicographical 
    # base orders). "lempel_ziv_0" stands for the (re-computed) 
    # original Lempel-Ziv data, which can be used to check whether it 
    # indeed is identical to the original data (which would indicate 
    # that the expressions are in the right order in the dataframe).
    perm_data["lempel_ziv_mean"] = perm_data[
    	["lempel_ziv_0", "lempel_ziv_1", "lempel_ziv_2"]
    ].mean(axis = 1)
    print("\nPERM_DATA\n", perm_data)

    # Add the new (perm) data to the original data.
    concat_data = pd.concat([lang_gen.big_data_table, perm_data], axis=1)
    print("\nORIGINAL DATA\n", lang_gen.big_data_table)
    print("\nCONCAT\n", concat_data, "\n")
    check = lang_gen.big_data_table["lempel_ziv"] == perm_data["lempel_ziv_0"]
    print("\nCHECK = lempel-ziv == lempel_ziv_0 =", check.all(), "\n")

    # Store concatenad new and old data as csv.
    utils.store_language_data_to_csv(
        concat_data, args.max_model_size, args.max_expr_len, 
        args.language_name, args.lang_gen_date, verbose=True
    )
    
    print("\nEND =", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    finish = time.time()
    print("\nrunning time =", (finish-start) / 60, "\n")

    

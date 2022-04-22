# Quantifiers satisfying semantic universals have shorter minimal description length

This repository accompanies Section 4 (Experiment 2: Language of Minimal Expression Length) in the following paper:
* Iris van de Pol, Paul Lodder, Leendert van Maanen, Shane Steinert-Threlkeld, and Jakub Szymanik, *Quantifiers satisfying semantic universals have shorter minimal description length*, submitted.

This project explores the complexity of quantifiers in the explanation of semantic universals, and compares these to earlier results on complexity (see [Van de Pol, Steinert-Threlkeld, Szymanik](https://cogsci.mindmodeling.org/2019/papers/0507/0507.pdf)) and learnability (see [Steinert-Threlkeld and Szymanik](https://semprag.org/index.php/sp/article/viewFile/sp.12.4/pdf)) in the explanation of semantic universals. In particular, we generate a large collection of quantifier expressions, based on a simple yet expressive grammar, and compute their complexities for two measures of complexity—minimal expression length and Lempel-Ziv (LZ) complexity—and whether they adhere to the universal properties of monotonicity, quantity, or conservativity.

We find that the LZ complexity results do not scale up robustly with respect to earlier findings, and we find that in terms of minimal expression length, quantifiers satisfying semantic universals are less complex: they have a shorter minimal description length. These results suggest that the simplicity of quantifier meanings, in terms of their minimal description length, partially explains the presence of semantic universals in the domain of quantifiers.

This repository contains all the code needed to replicate the data reported in that section. It also contains the data and figures that are reported therein.

If you have any questions and/or want to extend the code base and/or generate your own quantifier complexity data, feel free to get in touch!

## Getting Started

### Requirements

For install requirements see `requirements.txt` (use python 3.7).

This repository uses [dotenv](https://pypi.org/project/python-dotenv/).
Make sure to set `PROJECT_DIR` in `.env` to the location of the code in this repository on your local machine.

### Generating data

In order to **generate** quantifier expressions for a given set of language settings, use [src/generate_expressions.py](src/generate_expressions.py).

Example of a run from the command-line:

```
python src/generate_expressions.py  --max_expr_len 5 --max_model_size 8 --language_name "Logical_index"
```

The results will be stored under `PROJECT_DIR/results/` by default, but
this behaviour can be changed by setting `RESULTS_DIR_RELATIVE` in `.env`.

The `language_name` parameter refers to a `.json` file with language settings in the `language_setups` folder.
This file defines a name for the settings, the number of subsets that are used for repesenting quantifier models, and the collection of operators which defines the grammar of the language.
The number of subsets refers to the subareas of a quantifier model.
A model is defined by an ordered domain M, and two (possibly overlapping) subsets of M: A and B.
This gives four "subsets" which refer to the areas "AnotB", "AandB", "BnotA", and "neither".
Parameter `number_of_subsets` = 4 refers to all areas and `number_of_subsets` = 3 refers to the areas "AnotB", "AandB", and "BnotA".
The code is currently only guaranteed to work for `number_of_subsets` = 3.
For `number_of_subsets` = 1, 2, or 4 to work, some adjustments might need to be made to class method `generate_universe()` in [src/languagegenerator.py](src/languagegenerator.py).

The program [src/generate_expressions.py](src/generate_expressions.py) creates a LanguageGenerator object for a given `max_model_size` and `language_name`, as defined in [src/languagegenerator.py](src/languagegenerator.py).
Then by using the LanguageGenerator class method `gen_all_expr_and_their_scores()` for a given `max_expr_len`, a recursive procedure generates all (semantically unique) expressions of minimal expression length up to and including `max_expr_len`.
For each of those expressions their adherence to universal properties and their complexity scores are computed.
The LanguageGenerator object is stored in a `.dill` file, and the expressions and their scores are stored in a `.csv` file.
For Language=Logical this `.dill` file can be found in the zipped file [language_generator_up_to_length_7.dill.zip](results/Language=Logical-max_model_size=8/2022-03-11/language_generators/language_generator_up_to_length_7.dill.zip), and for Language=Logical_index a zipped `.dill` file can be downloaded at [the Open Science Framework](https://osf.io/nh9tw/).

Then the program [src/lex_perm.py](src/lex_perm.py) should be run to create two permutations of the binary quantifier representations (also called their meaning or extension) based on different lexicographical base orders over the quantifier models. The program computes the Lempel-Ziv complexities over these permuted quantifier representations.
See Section 4.1.3. *(Encoding Quantifier Meanings as Binary Sequences)* for more explanation about different lexicographical orders over the quantifier models.
The program loads the LanguageGenerator object stored in a `.dill` file to access the original quantifier representations and it stores the original data from the LanguageGenerator object (the expressions and their scores) plus the different Lempel-Ziv complexities over the permutations in a `.csv` file.

Before analyzing the data, use [src/adjust_csv.py](src/adjust_csv.py) for some post hoc additions to the `.csv` file with the language data (changing graded scores into binary, adding a score for having all three properties, mon_quan_cons, and adding standardized and randomly shuffled scores).

### Analyzing and plotting the data

To produce the descriptive results reported in the paper use [src/descriptive_stats.py](src/descriptive_stats.py).
This program prints the average complexity of all expressions with versus without a given universal property.
It also makes line plots of the percentage of expressions with a given universal property, plotted against minimal expression length and stores this as a `.pdf` file.
In addition, the program prints all quantifier expressions of length two that do not satisfy one or more of the universal properties, and it prints a contingency table for each universal property and minimal expression length (showing the frequency distributions for those two variables).

To produce the logistic regression plots reported in the paper use [src/distplot.py](src/distplot.py) with the following run from the command line:

```
python distplot.py --max_expr_len 5 --max_model_size 8 --language_name "Logical_index" --lang_gen_date "2020-12-25" --log_reg_date "2021-05-05" --sample_size 5000 --repeat 20000 --bootstrap_id 1
```

This will plot the distribution of the complexity coefficients for a bootstrapped logistic regression series with 20,000 runs, and sample size 5000, which are stored in `.csv` files in [results/Language=Logical_index-max_model_size=8/2020-12-25/analysis/log_regression/2021-05-05/csv/](results/Language=Logical_index-max_model_size=8/2020-12-25/analysis/log_regression/2021-05-05/csv/).
It also prints and stores the mean values and the 95% CI of the coefficients in a `.txt` file.

To run your own logistic regression series and store the results in `.csv` files, use [src/logistic_regression.py](src/logistic_regression.py).

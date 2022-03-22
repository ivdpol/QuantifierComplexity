''' 
This file is part of QuantifierComplexity.
'''
import argparse
import os
from pathlib import Path
import dotenv
import matplotlib.pyplot as plt  
import matplotlib.lines as mlines
import pandas as pd
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


def make_contingency_table(
    data: pd.DataFrame, x_var: str, y_var: str, verbose=True
) -> pd.DataFrame:
    '''Compute the frequency distributions of x_var vs y_var.

    Args:
        data: A DataFrame.
        x_var: A string, a column_name in data.
        y_var: A string, a column_name in data: needs to be a variable
            with possible values in {0,1}.
        verbose: True or False. Print the contingency table when True.
        
    Returns: A DataFrame with a contingency table for x_var vsf y_var.

    '''
    contingency_table = pd.crosstab(
        data[x_var], data[y_var], margins=True, margins_name="total"
    ) 
    # Per y_var value: percentage of #datapoints x_var=1 of 
    # total #datapoints.
    contingency_table["(" + y_var[0:4] + "=1)/total"] = \
        contingency_table[1] / contingency_table["total"]
    if verbose:
        print(contingency_table, "\n")
    return contingency_table


def plot_perc_with_prop_per_expr_len(
    quan_props: list, data: pd.DataFrame, max_model_size: int, 
    max_expr_len: int, language_name: str, lang_gen_date: str
):
    '''Plot percentage of satisfying expressions per expr length. 

    Print contingency table for each prop in quan_props.
    Store lineplot: percentage of quantifiers satisfying univ prop of
    all quantifiers per expression length. y := percentage, 
    x := minimal expression length.
    
    Args: 
        quan_props: A list of strings. The names of the properties.
        data: A pandas DataFrame with language data. Includes column
            names as given in quan_props.
        max_model_size: An int. Should coincide with the value in 
            max_model_size column in data.
        max_expr_len: An int. Should coincide with the max value of
            expr_length column in data.
        language_name: A string. Should coincide with the value of the 
            lot column in data.
        lang_gen_date: A string. Should coincide with the name of the 
            folder in which the csv folder with the csv file is present.
            The date on which the data was generated.

    '''
    contingency_tables = {prop: None for prop in quan_props}
    for prop in quan_props:  
        contingency_table = make_contingency_table(
            data, "expr_length", prop, verbose=True
        )
        contingency_tables[prop] = contingency_table
    fig, axs = plt.subplots(1, constrained_layout=True)
    markers = [".", "s", "o", "^"]
    # https://matplotlib.org/stable/api/markers_api.html
    colors = ["#e66101", "#fdb863", "#5e3c99", "#b2abd2"]
    # Colorbrew colors with diverging colorscheme:
    # colorblind safe, print friendly, photocopy safe.
    # From https://colorbrewer2.org/#type=diverging&scheme=PuOr&n=4.
    # "#e66101" = dark orange, "#fdb863" = light orange 
    # "#b2abd2" = light purple, "#5e3c99" = dark purple
    color_idx = 0
    for prop in quan_props:
        color = colors[color_idx]
        marker = markers[color_idx]
        contingency_tables[prop].drop("total", inplace=True)
        axs.scatter(
            x=contingency_tables[prop].index, 
            y=contingency_tables[prop]["(" + prop[0:4] + "=1)/total"],
            color=color,
            marker=marker
        )
        axs.plot(
            contingency_tables[prop].index, 
            contingency_tables[prop]["(" + prop[0:4] + "=1)/total"],
            color=color
        )
        axs.set_xlabel("Minimal expression length", fontsize=16)
        axs.set_xticks(range(1, args.max_expr_len + 1))
        color_idx += 1
    axs.set_ylabel("% Quantifiers with universal property", fontsize=16)
    # Make legend shapes into rectangles, see:
    # https://stackoverflow.com/questions/40672088/matplotlib
    # -customize-the-legend-to-show-squares-instead-of-rectangles
    plt.rcParams['legend.handlelength'] = 1
    plt.rcParams['legend.handleheight'] = 1.125
    if len(quan_props) == 4:
        legend_labels = [
            "Monotonicity", "Quantity", "Conservativity", "All three properties"
        ]
    if len(quan_props) == 3:
        legend_labels = [
            "Monotonicity", "Conservativity", "Both properties"
        ]
    legend_handles = []
    for i, _ in enumerate(quan_props):
        legend_handles.append(
            mlines.Line2D([], [],
                color=colors[i],
                marker=markers[i],
                label=legend_labels[i]
            )
        )
    plt.legend(handles=legend_handles, fontsize=16)
    file_loc = utils.make_descriptive_stats_path( 
        max_model_size, language_name, lang_gen_date
    )
    basic_filename = utils.make_basic_filename(
            max_model_size, max_expr_len, language_name  
    )
    plt.savefig(
        file_loc / f"percentage_and_avg_{basic_filename}.pdf", format="pdf"
    )
    plt.close()


def avg_compl_with_vs_without(score: str):
    '''Print the mean score for quans with vs without univ prop.

    First print the means for raw values, then the means for zscores.
    Print lines suitable for inserting in LaTeX table.

    Args:
        Score: A String. Either "ml" (minimal expression length),
            or "lz" (Lempel Ziv).

    '''
    measures = {
        "ml": [("expr_length", "raw scores"), ("ml_zscore", "zscores")], 
        "lz_0": [("lempel_ziv_0","raw scores"), ("lz_0_zscore", "zscores")],
        "lz_1": [("lempel_ziv_1","raw scores"), ("lz_1_zscore", "zscores")],
        "lz_2": [("lempel_ziv_2","raw scores"), ("lz_2_zscore", "zscores")],
        "lz_mean": [("lempel_ziv_mean","raw scores"), ("lz_mean_zscore", "zscores")]
    }
    line30 = "-" * 30
    print(line30)
    print(measures[score][0][0])
    for measure in measures[score]:
        print(f"{line30}\n{measure[1]}\n{line30}")
        for quan_prop in quan_props:
            avg_with = round(
                data.loc[data[quan_prop] == 1][measure[0]].mean(), 2
            )
            avg_without = round(
                data.loc[data[quan_prop] == 0][measure[0]].mean(), 2
            )
            proportion = round(
                data.loc[data[quan_prop] == 1].shape[0] / data.shape[0], 2
                )
            print(
                f"{quan_prop} & {avg_with} & {avg_without}",
                f"& {proportion}  \\\\"
                #& {average} \\\\"
            )
        print()


def expressions_non_satisfying(expr_len: int, data: pd.DataFrame):
    '''Print expressions of given len that don't satisfy univ props.
    
    Args:
        expr_len: An int. Should be a value of expr_length column  
            in data. 
        data: A pandas DataFrame with language data.

    '''
    line = "-" * 60
    print(
        f"\n{line}", "\nNon-satisfying expressions of expr-length 2\t", 
        f"\n{line}\n",
    )
    data.sort_values("expression", inplace=True)
    # Non-monotonic.
    data_non_mon = data.loc[
        (data["expr_length"] == expr_len) & (data["monotonicity"] == 0)
    ]
    print("non-monotonicity\t", len(data_non_mon.index))
    print(data_non_mon[["expression"]], "\n")
    # Non-quantitative.
    data_non_quan = data.loc[
        (data["expr_length"] == expr_len) & (data["quantity"] == 0)
    ]
    print("non-quantitative\t", len(data_non_quan.index))
    print(data_non_quan[["expression"]], "\n")
    # Non-Conservative.
    data_non_cons = data.loc[
        (data["expr_length"] == expr_len) & (data["conservativity"] == 0)
    ]
    print("non-conservative\t", len(data_non_cons.index))
    print(data_non_cons[["expression"]], "\n")
    # Satisfying none (non-mon, non-quan, and non-cons).
    data_none = data.loc[
        (data["expr_length"] == expr_len) & (data["conservativity"] == 0) & 
        (data["monotonicity"] == 0) & (data["quantity"] == 0)
    ]
    print("none\t", len(data_none.index))
    print(data_none[["expression"]], "\n")


if __name__ == "__main__":
    # Default values for argparse args.
    LANGUAGE_NAME = "Logical"       # "Logical_index"       # "Logical"
    MAX_EXPR_LEN = 7                # 5 for Logical_index   # 7 for Logical
    MAX_MODEL_SIZE = 8 
    LANG_GEN_DATE = "2022-03-11"    # "2022-03-11"  # "2020-12-25" 
    CSV_DATE = "2022-03-16"         # "2022-03-16"  # "2021-05-05 for Logical_index
                                                    # "2021-03-23" for Logical     
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

    print(data)

    if "index" in args.language_name:
        quan_props = [
            "monotonicity", "quantity", "conservativity", "mon_quan_cons"
        ]
    else: 
        quan_props = ["monotonicity", "conservativity", "mon_cons"]

    # Show expressions of lenght 2 non-satisfying univ props.
    expressions_non_satisfying(2, data)

    # Make contigency tables and plots of percentage of quans with univ
    # prop, per expr length.
    line = "-" * 60
    print(f"\n{line}\nContingency tables\n{line}\n")
    plot_perc_with_prop_per_expr_len(
        quan_props, data, args.max_model_size, args.max_expr_len, 
        args.language_name, args.lang_gen_date
    )

    # Print avg complexity quans with versus without univ prop.
    # Print suitable for latex table.
    print(
        f"\n{line}\nAverage complexity expression with vs. without univ prop",
        f"\nproportion of quans with univ prop\n{line}\n",
    )
    avg_compl_with_vs_without("ml")
    avg_compl_with_vs_without("lz_0")
    avg_compl_with_vs_without("lz_1")
    avg_compl_with_vs_without("lz_2")
    avg_compl_with_vs_without("lz_mean")
    print(f"{line}\nMax and avg\n{line}")
    print("max_ml\t\t", data["expr_length"].max())
    line30 = "-" * 30
    print("avg_ml\t\t", round(data["expr_length"].mean(), 2), f"\n{line30}")
    print("max_lz\t\t", round(data["lempel_ziv"].max(), 2))
    print("avg_lz\t\t", round(data["lempel_ziv"].mean(), 2), "\n")
    print("max_lz_1\t", round(data["lempel_ziv_1"].max(), 2))
    print("avg_lz_1\t", round(data["lempel_ziv_1"].mean(), 2), "\n")
    print("max_lz_2\t", round(data["lempel_ziv_2"].max(), 2))
    print("avg_lz_2\t", round(data["lempel_ziv_2"].mean(), 2), "\n")
    print("max_lz_m\t", round(data["lempel_ziv_mean"].max(), 2))
    print("avg_lz_m\t", round(data["lempel_ziv_mean"].mean(), 2), "\n")

# 5
# LZ_mean
# monotonicity & -0.22 & 0.08 & 0.28  \\
# quantity & 0.02 & -0.0 & 0.15  \\
# conservativity & 0.16 & -0.02 & 0.12  \\
# mon_quan_cons & 0.16 & -0.0 & 0.02  \\
# LZ_0
# monotonicity & -0.19 & 0.08 & 0.28  \\
# quantity & 0.03 & -0.0 & 0.15  \\
# conservativity & 0.18 & -0.02 & 0.12  \\
# mon_quan_cons & 0.16 & -0.0 & 0.02  \\
# LZ_1
# monotonicity & -0.23 & 0.09 & 0.28  \\
# quantity & 0.0 & -0.0 & 0.15  \\
# conservativity & 0.19 & -0.02 & 0.12  \\
# mon_quan_cons & 0.16 & -0.0 & 0.02  \\
# LZ_2
# monotonicity & -0.23 & 0.09 & 0.28  \\
# quantity & 0.03 & -0.0 & 0.15  \\
# conservativity & 0.12 & -0.02 & 0.12  \\
# mon_quan_cons & 0.15 & -0.0 & 0.02  \\

# 7
# LZ_mean
# monotonicity & -0.05 & 0.03 & 0.33  \\
# conservativity & 0.34 & -0.06 & 0.14  \\
# mon_cons & 0.22 & -0.02 & 0.08  \\
# LZ_0
# monotonicity & -0.05 & 0.02 & 0.33  \\
# conservativity & 0.34 & -0.06 & 0.14  \\
# mon_cons & 0.22 & -0.02 & 0.08  \\
# LZ_1
# monotonicity & -0.05 & 0.03 & 0.33  \\
# conservativity & 0.37 & -0.06 & 0.14  \\
# mon_cons & 0.24 & -0.02 & 0.08  \\
# LZ_2
# monotonicity & -0.05 & 0.03 & 0.33  \\
# conservativity & 0.33 & -0.06 & 0.14  \\
# mon_cons & 0.19 & -0.02 & 0.08  \\
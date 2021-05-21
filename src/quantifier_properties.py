'''
This file is part of QuantifierComplexity.
'''
import time
from math import log2
import pandas as pd
import numpy as np
import networkx as nx
from lempel_ziv_complexity import lempel_ziv_complexity as lz


class QuantifierProperty:
    '''Base class for various quantifier properties.'''
    pass


class Uniformity(QuantifierProperty):
    """Class to collect static methods related to uniformity.

    No objects will be created for this class.

    """
    name = "uniformity"

    @staticmethod
    def property_function(bin_str: np.array) -> float:
        """
        Return the uniformity of a binary string.
        
        Uniformity = max between ratio of 1's and ratio of 0's 
        of bin str.

        Args:
            bin_str: A numpy array of shape (1,) with Boolean values.
                Represents the meaning (extension) of an expression.
                Entry i represents whether the expression is true
                in model i (the i'th model in the enumeration of 
                models in the universe).

        Returns:
            A single number (float).

        """
        expr_len = len(bin_str)
        # #1's(bin_str) / len(bin_str).
        ratio_1 = bin_str.sum() / expr_len 
        # #0's(bin_str) / len(bin_str).
        ratio_0 = np.invert(bin_str).sum() / expr_len 
        return max(ratio_1, ratio_0)
        

class LempelZiv(QuantifierProperty):
    """Class to collect static methods related to LempelZiv complexity.

    No objects will be created for this class.

    """
    name = "lempel_ziv"

    @staticmethod
    def property_function(bin_str: np.array) -> float:
        """Compute measure based on Lempel_Ziv complexity. 

        Same measure as in Dingle, Camargo, and Louis (2018),
        'Input–output maps are strongly biased towards simple outputs', 
        See Supplementary Note 1, p. 10.

        Args:
            bin_str: A numpy array of shape (1,) with Boolean values.
                Represents the meaning (extension) of an expression.
                Entry i represents whether the expression is true
                in model i (the i'th model in the enumeration of 
                models in the universe).

        Returns:
            A single number (float).

        """
        expr_len = len(bin_str)
        if bin_str.all() or np.invert(bin_str).all():
            return log2(expr_len)
        lz_expr = lz(bin_str.tobytes())
        lz_rev_expr = lz(np.flip(bin_str).tobytes())
        return log2(expr_len) * (lz_expr + lz_rev_expr) / 2


class Monotonicity(QuantifierProperty):
    """Class to collect static methods related to monotonicity.

    No objects will be created for this class.

    """
    name = "monotonicity"

    @staticmethod
    def signature_function_up(meaning_matrix: pd.DataFrame) -> pd.DataFrame:
        """Return the signature values for upward monotonicity.

        Given a model and an expression, the signature value is True 
        if the model has a *submodel* that makes the expression true,
        and False otherwise.

        Args:
            meaning_matrix: A pandas Dataframe of which the index values 
                are all models up to max_model_size and the column  
                values are all expressions up to max_expression_length. 
                Cel (i,j) shows whether model i makes expression j true 
                (1) or not (0).

        Returns:
            A pandas Dataframe with boolean values, which are the upward 
            monotonicity signature values per (model,expression) pair.
            Cel (i,j) shows the monotonicity signature value for model i 
            and expression j.

        """
        return Monotonicity.signature_function_generic(
            meaning_matrix, "upwards"
        )

    @staticmethod
    def signature_function_down(meaning_matrix: pd.DataFrame) -> pd.DataFrame:
        """Return the signature values for downward monotonicity.

        Given a model and an expression, the signature value is True 
        if the model has a *supermodel* that makes the expression true,
        and False otherwise.

        Args:
            meaning_matrix: A pandas Dataframe of which the index 
                values are all models up to max_model_size and the 
                column values  are all expressions up to 
                max_expression_length. Cel (i,j) shows whether model 
                i makes expression j true (1) or not (0).

        Returns:
            A pandas Dataframe with boolean values, which are the 
            downward monotonicity signature values per 
            (model,expression) pair. Cel (i,j) shows the monotonicity 
            signature value for model i and expression j.

        """
        return Monotonicity.signature_function_generic(
            meaning_matrix, "downwards"
        )

    @staticmethod
    def signature_function_generic(
        meaning_matrix: pd.DataFrame, monotonicity_type: str, verbose=False
    ) -> pd.DataFrame:
        """ Return signature values for upward or downward monotonicity.
        
        Given a model and an expression, the signature value is True 
        if the model has a submodel (upward mon) or supermodel 
        (downward mon) that makes the expression true, and False 
        otherwise.
        
        Args:
            meaning_matrix: A pandas Dataframe of which the index values 
                are all models up to max_model_size and the column 
                values are all expressions up to max_expression_length. 
                Cel (i,j) shows whether model i makes expression j 
                true (1) or not (0). 
            monotonicity_type: A string, either "upwards" or "downwards. 
            verbose: True or False. Print computing time of making 
                submodels when True.

        Returns:
            A pandas Dataframe with boolean values, which are the upward 
            or downward monotonicity signature values per 
            (model,expression) pair. Cel (i,j) shows the monotonicity 
            signature value for model i and expression j.
        
        """
        # Make graph that captures sub/supermodel relations
        start = time.time()
        model_relation_graph = _make_sub_supermodel_graph(
            meaning_matrix.index.values
        )
        finish = time.time()
        if verbose:
            print("making submodles", finish-start)
        # Convert binary matrix to bool to speed computations
        meaning_matrix = meaning_matrix.astype(bool)
        # Temporarily convert index names (tuples) into strings,
        # to avoid multilevel column problems later on with 
        # exp2true_in_any. Save the original index names to later 
        # put back.
        tuple_models = dict()
        for model in meaning_matrix.index:
            tuple_models[str(model)] = model
        meaning_matrix.index = [str(model) for model in meaning_matrix.index]
        # Make DataFrame with index = expressions, columns = models,
        # to fill with the monotonicity signature values.
        # For monotonicity_type = "upwards": 
        # The pair (expression,model) has a true sub_super_model, 
        # when the model has a *submodel* that makes the espression 
        # true. For monotonicity_type == "downwards": 
        # The pair (expression,model) has a true sub_super_model, 
        # when the model has a *supermodel* that makes the spression 
        # true.
        expr_model2has_true_sub_super_model = pd.DataFrame(
            np.zeros_like(meaning_matrix.T),
            columns=meaning_matrix.index.values,
            index=meaning_matrix.columns,
        )
        # Compute all has_true_sub_or_super_model values,
        # and fill in expr_model2has_true_sub_or_super_model.
        for model in tuple_models.values():
            # Get all sub-/supermodels for this model, as strings,
            # to avoid multi-level column issues with exp2true_in_any.
            if monotonicity_type == "upwards": 
                # Submodels case.
                sub_super_models = [model] + list(
                    nx.ancestors(model_relation_graph, model)
                )
            elif monotonicity_type == "downwards": 
                # Supermodels case.
                sub_super_models = [model] + list(nx.descendants(
                    model_relation_graph, model)
                )
            sub_super_models_strings = [
                str(sub_super_model) for sub_super_model in sub_super_models
            ]
            # If any of the sub-/supermodels of this model satisfy a  
            # given expression, then the signature value for that 
            # expression is True.
            exp2true_in_any = meaning_matrix.loc[
                sub_super_models_strings
            ].any(0)
            expr_model2has_true_sub_super_model[str(model)] = exp2true_in_any
        # Transpose to get a matrix with models as index and  
        # expressions as column.
        signatures = expr_model2has_true_sub_super_model.T
        # Replace column names strings by their original tuple.
        meaning_matrix.rename(index=tuple_models, inplace=True)
        signatures.rename(index=tuple_models, inplace=True)
        return signatures


class Quantity(QuantifierProperty):
    """Class to collect static methods related to quantity.

    No objects will be created for this class.

    """
    name = "quantity"

    @staticmethod
    def signature_function(meaning_matrix: pd.DataFrame) -> pd.Series:
        """Returns quantity signature value for each model.

        The signature value for a model (A, B, <) is 
        (|A intersect B|, |A \ B|, |B \ A|).

        Args:
            meaning_matrix: A pandas Dataframe of which the index values 
                are all models up to max_model_size and the column 
                values are all expressions up to max_expression_length. 
                Cel (i,j) shows whether model i makes expression j 
                true (1) or not (0).

        Returns: A pd.Series with a signature value for each model. 
            The signature value at index i is the signature value
            for the i-th model in the enumeration of models 
            in meaning-matrix.

        """
        out = []
        # Each model is a list with tuples. Each tuple represents 
        # one object, of which index 0 shows membership in A and index 1
        # and index 1 shows membership in B.
        for model in meaning_matrix.index.values:
            signature = [0, 0, 0]
            for (object_in_a, object_in_b) in model:
                if object_in_a and object_in_b:
                    signature[0] += 1
                elif object_in_a:
                    signature[1] += 1
                elif object_in_b:
                    signature[2] += 1
            out.append(tuple(signature))
        return pd.Series(out)


class Conservativity(QuantifierProperty):
    """Class to collect static methods related to concervativity.

    No objects will be created for this class.

    """
    name = "conservativity"

    @staticmethod
    def signature_function(meaning_matrix: pd.DataFrame) -> pd.Series:
        """Returns conservativity signature value for each model.

        The signature value for a model (A,B,<) is (A, A min B,<). 
        (A, A min B,<) is represented by a tuple with 0's, and 1's.
        0 at index i means: the i-th object is in A min B.
        1 at index i means: the i-th object is in A.

        Args:
            meaning_matrix: A pandas Dataframe of which the index values 
                are all models up to max_model_size and the column 
                values are all expressions up to max_expression_length. 
                Cel (i,j) shows whether model i makes expression 
                j true (1) or not (0).

        Returns:
            A pd.Series with a signature value for each model. 
            The signature value at index i is the signature value
            for the i-th model in the enumeration of models 
            in meaning-matrix.

        """
        out = []
        # Each model is a list with tuples. Each tuple represents 
        # one object, of which index 0 shows membership in A and index 1
        # shows membership in B.
        for model in meaning_matrix.index:
            signature = []
            for (object_in_a, object_in_b) in model:
                # Object in A min B.
                if object_in_a and not object_in_b:
                    signature.append(0)
                # Object in A.
                if object_in_a and object_in_b:
                    signature.append(1)
            out.append(tuple(signature))
        return pd.Series(out)


def _make_sub_supermodel_graph(models: list) -> nx.DiGraph:
    """Make a graph with the supermodel relationships between models.

    Definition supermodel (using set notation): 
    let model m0 = < A,B,R >, and model m1 = < A',B',R' >, then m1 is 
    a supermodel of m0 iff A = A', B \se B', and R \se R', i.e. the 
    relative ordering of the elements in A and B is presevered in A' 
    and B'. So when going from a model to a supermodel, the only 
    operations that are allowed, are to (1) add elements the area BnotA, 
    or (2) move objects from AnotB to AandB.

    Definition submodel: Model m1 is a submodel of model m0 iff m0 
    is a supermodel of m1. 

    Args:
        models: A list of quantifier models, which are tuples 
            that contain model-objects, which are pairs (tuples) 
            of the form (1,1), (1,0), or (1,0). Entries in 
            index 0 represent membership in A, and entries 
            in index 1 represent membership in B.

    Returns: A directed graph with one node for each model.
        When node_x -> node_y, then node_y is a supermodel 
        of node_x, and node_x is a submodel of node_y. 
        nx.descendants(graph,model_x) will return all supermodels 
        of model_x, and nx.ancestors(graph,model_x) will return all 
        submodels of model_x.

    """
    max_model_size = len(models[-1])
    edges = []
    # For each model, add an edge (model,supermodel), for each
    # supermodel that can be obtained by changing model by one element.
    # Allowed changes are: (1) add (0,1) at a position before, between, 
    # or after the elements of model; and (2) change an element of model
    # from (1,0) to (1,1).
    for model in models:
        # # Ech model is a supermodel of itself
        # edges.append((model,model)) 
        for idx, _ in enumerate(model):
            # Change an element of model from (1,0) to (1,1).
            if model[idx] == (1, 0):
                edges.append((model, model[:idx]+((1, 1),)+model[idx+1:]))
            # Only models with length < max_model_size have a supermodel
            # that is longer, because max_size of any supermodel =
            # max_model_size
            if len(model) < max_model_size:
                # Add an element (0,1) before the end of the model
                edges.append((model, model[:idx]+((0, 1),)+model[idx:]))
                # Add an element (0,1) at the the end of the model.
                if idx == len(model)-1:
                    edges.append((model, model+((0, 1),)))
    graph = nx.DiGraph()
    graph.add_nodes_from(models)
    graph.add_edges_from(edges)
    return graph

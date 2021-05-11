'''
This file is part of QuantifierComplexity.
'''

def pre_filter(lang_gen, expr: tuple):
    '''Filter out expressions based on syntax.

    Return whether the given expression has the same meaning as a shorter
    equivalent expression, based on several syntactic checks. If the function
    returns True, the expression can safely be disregarded. If the function
    returns False, there might still be a shorter equivalent expression that
    the (incomplete) syntactic check does not detect.
    
    '''
    # Manually add a tautology to the language. Other tautologies
    # are filtered out.
    if expr == (">", "1", "0"):
        return False
    # Manually add a falsehood to the language. Other falsehoods
    # are filtered out.
    if expr == (">", "0", "0"):
        return False
    return "filter" in pre_filter_recursive(lang_gen, expr)


def pre_filter_recursive(lang_gen, expr: tuple):
    '''Recursive auxiliary funcion for pre_filter().

    Return a list of labels that carry some information about the meaning of
    the given expression (based on the labels of subexpressions). These labels
    are then used to detect whether an expression can safely be filtered out
    because there is a shorter equivalent expression.

    The following labels with the following meaning are used:
    - "filter": the expression can safely be filtered out
    - "subsetOfA": the expression has as meaning a subset of the set A
    - "subsetOfB": the expression has as meaning a subset of the set B
    - "supersetOfA": the expression has as meaning a superset of the set A
    - "supersetOfB": the expression has as meaning a superset of the set B
    - "primary": the expression has as meaning one of the 'primary' sets (i.e.
      sets that can be formed from A and B using basic set operations)
    - "isDigit": the expression is a number constant
    - "atmostsingleton": the expression has as meaning a set of size at most 1
    - "combinedset": the expression has as main operator one of the set
      theoretic operators (union, intersection, difference)

    '''
    # For atomic expressions, return list of appropriate labels.
    if expr == "A":
        return ["subsetOfA", "supersetOfA", "primary"]
    if expr == "B":
        return ["subsetOfB", "supersetOfB", "primary"]
    if str(expr).isdigit():
        return ["isDigit"]

    main_operator = expr[0]

    # Filter out some handpicked duplicates with smaller equivalent
    # expressions.
    if expr == ('>', ('card', ('union', 'A', 'B')), '0'):
        return ["filter"]

    # For primary areas (except the empty set), return list of
    # appropriate labels.
    if expr == ("union", "A", "B"):
        return ["primary", "supersetOfA", "supersetOfB"]
    if expr == ("diff", "A", "B"):
        return ["primary", "subsetOfA"]
    if "intersection" in lang_gen.operators:
        if expr == ("intersection", "A", "B"):
            return ["primary", "subsetOfA", "subsetOfB"]
    else:
        if expr == ("diff", "A", ("diff", "A", "B")):
            return ["primary", "subsetOfA", "subsetOfB"]
    if expr == ("diff", "B", "A"):
        return ["primary", "subsetOfB"]
    if expr == ("union", ("diff", "A", "B"), ("diff", "B", "A")):
        return ["primary"]

    # Recursively compute the labels of subexpressions.
    info_tuple = tuple(map(pre_filter_recursive, [lang_gen] * 2, expr[1:]))

    # If any subexpression contains the label "filter" (meaning the
    # expression should be filtered), return the list with just the
    # label "filter".
    for info in info_tuple:
        if "filter" in info:
            return ["filter"]

    # Comparisons with two constant numbers are always either true
    # or false, so we can filter them.
    if len(expr) >= 3 and str(expr[1]).isdigit() and str(expr[2]).isdigit():
        return ["filter"]
    # Filter expressions of the form (0 > X), because equivalent
    # to falsum.
    if main_operator == ">" and str(expr[1]) == "0":
        return ["filter"]
    # Filter expressions of the form (X > max_model_size), because
    # equivalent to falsum.
    if main_operator == ">" and str(expr[2]) == str(lang_gen.max_model_size):
        return ["filter"]
    # If the two subexpressions are the same, for these operators,
    # we can filter the entire expression.
    if (len(expr) >= 3 and main_operator in [
            # A \ A is the emptyset, which is never present in minimal
            # expressions, therefore filtered.
            "subset", "union", "intersection", "diff", ">", "="
    ] and str(expr[1]) == str(expr[2])):
        return ["filter"]

    # For expressions formed with the = operator:
    if main_operator == "=":
        # For number comparisons with "=", only allow constants on
        # the right hand side.
        if str(expr[1]).isdigit():
            return ["filter"]

        # For number comparisons with "=", the cardinality of a
        # singleton (with constant) and a constant, there always
        # exists another shorter expression.
        if (expr[1][0] == "card" and
                # expr[1][1][0] == "index" and
                "atmostsingleton" in pre_filter_recursive(
                    lang_gen,
                    expr[1][1]
                ) and
                str(expr[1][1][1]).isdigit() and
                str(expr[2]).isdigit()):
            return ["filter"]

        # Forbid expressions of the form |i(...)| = 2+, because
        # equivalent to falsum.
        if (expr[1][0] == "card" and
                # expr[1][1][0] == "index" and
                "atmostsingleton" in pre_filter_recursive(
                    lang_gen,
                    expr[1][1]
                ) and
                str(expr[2]).isdigit() and
                int(str(expr[2])) >= 2):
            return ["filter"]

    # For expressions formed with the > operator:
    if main_operator == ">":
        # For number comparisons with ">", forbid "1" on the left
        # hand side (because that's equivalent to "= 0").
        if expr[1] == "1":
            return ["filter"]

        # For number comparisons with ">", the cardinality of a
        # singleton (with constant) and a constant, there always
        # exists another shorter expression.
        if (expr[1][0] == "card" and
                expr[1][1][0] == "index" and
                str(expr[1][1][1]).isdigit() and
                str(expr[2]).isdigit()):
            return ["filter"]
        if (expr[2][0] == "card" and
                expr[2][1][0] == "index" and
                str(expr[2][1][1]).isdigit() and
                str(expr[1]).isdigit()):
            return ["filter"]

        # Forbid expressions of the form |i(...)| > 1+, because
        # equivalent to falsum.
        if (expr[1][0] == "card" and
                # expr[1][1][0] == "index" and
                "atmostsingleton" in pre_filter_recursive(
                    lang_gen,
                    expr[1][1]
                ) and
                str(expr[2]).isdigit() and
                int(str(expr[2])) >= 1):
            return ["filter"]

        # Forbid expressions of the form 2+ > |i(...)|, because
        # equivalent to tautology.
        if (expr[2][0] == "card" and
                # expr[2][1][0] == "index" and
                "atmostsingleton" in pre_filter_recursive(
                    lang_gen,
                    expr[2][1]
                ) and
                str(expr[1]).isdigit() and
                int(str(expr[1])) >= 2):
            return ["filter"]

    # For number comparisons with ">", forbid (max_model_size-1)
    # on the right hand side (because that's equivalent to
    # "= max_model_size").
    if main_operator == ">":
        if expr[2] == str(lang_gen.max_model_size-1):
            return ["filter"]

    # We can filter out some combinations of Boolean combinations.
    if main_operator == "not":
        if str(expr[1]).startswith("('not"):
            return ["filter"]
        if (str(expr[1]).startswith("('>") and
                (str(expr[1][1]).isdigit() or
                 str(expr[1][2]).isdigit())):
            return["filter"]
    if (main_operator == "and" and
            str(expr[1]).startswith("('not") and
            str(expr[2]).startswith("('not")):
        return ["filter"]
    if (main_operator == "or" and
            str(expr[1]).startswith("('not") and
            str(expr[2]).startswith("('not")):
        return ["filter"]

    # For expressions formed with the subset operator:
    if main_operator == "subset":

        # Subsets of A are always a subset of supersets of A,
        # so expressions like this are always true (and thus we
        # can filter them out)
        if ("subsetOfA" in info_tuple[0] and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if ("subsetOfB" in info_tuple[0] and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]

        # Filter out expressions of the form X ⊆ i(2+,Y) where X is a
        # superset of A and Y is a subset of A, because equivalent
        # to falsum
        if ("supersetOfA" in info_tuple[0] and
                expr[1][0] == "index" and
                str(expr[1][1]).isdigit() and
                int(str(expr[1][1])) >= 2 and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]
        if ("supersetOfB" in info_tuple[0] and
                expr[1][0] == "index" and
                str(expr[1][1]).isdigit() and
                int(str(expr[1][1])) >= 2 and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]

        # Filter out expressions of the form: X ⊆ index(c,X) for
        # a constant c, because equivalent to |X| < 1, |X| < 2, or
        # to falsum (depending on the value of c).
        if (expr[2][0] == "index" and
                str(expr[2][1]).isdigit() and
                expr[1] == expr[2][2]):
            return ["filter"]

        # Filter out expressions of the form (X \ Y) ⊆ Z,
        # where Y is a subset of A and Z is a superset of A,
        # (because equivalent to X ⊆ Z), and similarly for B.
        if (expr[1][0] == "diff" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]

        # Filter out expressions of the form (X ∪ Y) ⊆ Z,
        # where Y is a subset of A and Z is a superset of A,
        # (because equivalent to X ⊆ Z), and similarly for B.
        if (expr[1][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]

        # Filter out expressions of the form X ⊆ (Y ∪ Z),
        # where X is a superset of A and Y is a subset of A,
        # (because equivalent to X ⊆ Z), and similarly for B.
        if (expr[2][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][1]) and
                "supersetOfA" in info_tuple[0]):
            return ["filter"]
        if (expr[2][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][2]) and
                "supersetOfA" in info_tuple[0]):
            return ["filter"]
        if (expr[2][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][1]) and
                "supersetOfB" in info_tuple[0]):
            return ["filter"]
        if (expr[2][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][2]) and
                "supersetOfB" in info_tuple[0]):
            return ["filter"]

        # Filter out expressions of the form X ⊆ (X ∩ Y)
        # (because equivalent to X ⊆ Y).
        if (expr[2][0] == "intersection" and
                expr[1] == expr[2][1]):
            return ["filter"]
        if (expr[2][0] == "intersection" and
                expr[1] == expr[2][2]):
            return ["filter"]

        # Filter out expressions of the form X ⊆ (X ∪ Y)
        # (because equivalent to T).
        if (expr[2][0] == "union" and
                expr[1] == expr[2][1]):
            return ["filter"]
        if (expr[2][0] == "union" and
                expr[1] == expr[2][2]):
            return ["filter"]

        # Filter out expressions of the form X ⊆ (Y \ X)
        # (because equivalent to |X| = 0).
        if (expr[2][0] == "diff" and
                expr[1] == expr[2][2]):
            return ["filter"]

        # Filter out expressions of the form (X ∩ Y) ⊆ X
        # (because equivalent to T).
        if (expr[1][0] == "intersection" and
                expr[2] == expr[1][1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                expr[2] == expr[1][2]):
            return ["filter"]

        # Filter out expressions of the form (X ∪ Y) ⊆ X
        # (because equivalent to Y ⊆ X).
        if (expr[1][0] == "union" and
                expr[2] == expr[1][1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                expr[2] == expr[1][2]):
            return ["filter"]

        # Filter out expressions of the form (X \ Y) ⊆ X
        # (because equivalent to T).
        if (expr[1][0] == "diff" and
                expr[2] == expr[1][1]):
            return ["filter"]

        # Filter out expressions of the form (Y \ X) ⊆ X
        # (because equivalent to |Y| = 0).
        if (expr[1][0] == "diff" and
                expr[2] == expr[1][2]):
            return ["filter"]

        # Filter out expressions of the form (X \ Y) ⊆ (X \ Z)
        # because equivalent to Z ⊆ Y.
        if (expr[1][0] == "diff" and
                expr[2][0] == "diff" and
                expr[1][1] == expr[2][1]):
            return ["filter"]

        # Filter out expressions of the form (X ∪ Y) ⊆ A
        # where X is a subset of A
        # (because equivalent to Y ⊆ A),
        # and similarly for B.
        if (expr[1][0] == "union" and
                expr[2] == "A" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][1])):
            return ["filter"]
        if (expr[1][0] == "union" and
                expr[2] == "A" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]
        if (expr[1][0] == "union" and
                expr[2] == "B" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][1])):
            return ["filter"]
        if (expr[1][0] == "union" and
                expr[2] == "B" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]

        # Filter out expressions of the form index(j,X) ⊆ X
        # (because equivalent to T).
        if (expr[1][0] == "index" and
                expr[2] == expr[1][2]):
            return ["filter"]

    # For expressions formed with the union operator:
    if main_operator == "union":
        # Label them with "combinedset" if they are not filtered out.
        info = ["combinedset"]
        # Filter out expressions that are equivalent to (A ∪ B)
        if ("subsetOfA" in info_tuple[0] and
                "supersetOfA" in info_tuple[0] and
                "subsetOfB" in info_tuple[1] and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if ("subsetOfA" in info_tuple[1] and
                "supersetOfA" in info_tuple[1] and
                "subsetOfB" in info_tuple[0] and
                "supersetOfB" in info_tuple[0]):
            return ["filter"]

        # Filter out expressions of the form (X ∪ Y) where X is a
        # superset of A and Y is a subset of A (because equivalent
        # to Y); and similarly for B.
        if "supersetOfA" in info_tuple[0] and "subsetOfA" in info_tuple[1]:
            return ["filter"]
        if "supersetOfA" in info_tuple[1] and "subsetOfA" in info_tuple[0]:
            return ["filter"]
        if "supersetOfB" in info_tuple[0] and "subsetOfB" in info_tuple[1]:
            return ["filter"]
        if "supersetOfB" in info_tuple[1] and "subsetOfB" in info_tuple[0]:
            return ["filter"]

        # Add the labels "subsetOfA"/"supersetOfA"/etc as needed
        if "supersetOfA" in info_tuple[0]:
            info.append("supersetOfA")
        if "supersetOfA" in info_tuple[1]:
            info.append("supersetOfA")
        if "supersetOfB" in info_tuple[0]:
            info.append("supersetOfB")
        if "supersetOfB" in info_tuple[1]:
            info.append("supersetOfB")
        if "subsetOfA" in info_tuple[0] and "subsetOfA" in info_tuple[1]:
            info.append("subsetOfA")
        if "subsetOfB" in info_tuple[0] and "subsetOfB" in info_tuple[1]:
            info.append("subsetOfB")

        # Adding a subset of A to a superset of A will just give the
        # superset, so we can filter these expressions (similarly for B)
        if "supersetOfA" in info_tuple[0] and "subsetOfA" in info_tuple[1]:
            return ["filter"]
        if "supersetOfA" in info_tuple[1] and "subsetOfA" in info_tuple[0]:
            return ["filter"]
        if "supersetOfB" in info_tuple[0] and "subsetOfB" in info_tuple[1]:
            return ["filter"]
        if "supersetOfB" in info_tuple[1] and "subsetOfB" in info_tuple[0]:
            return ["filter"]

        # We only allow primary sets on the left hand side
        if "primary" in info_tuple[1]:
            return ["filter"]

        # For joining combinedsets with singletons, only allow one order
        # (where the singleton is on the right hand side)
        if ("singleton" in info_tuple[0] and
                "combinedset" in info_tuple[1]):
            return ["filter"]

        # For joining combinedsets with combinedsets or
        # or joining singletons with singletons, only allow them
        # in alphabetic order (to remove duplicates).
        if ("combinedset" in info_tuple[0] and
                "combinedset" in info_tuple[1] and
                str(expr[1]) > str(expr[2])):
            return ["filter"]
        if ("singleton" in info_tuple[0] and
                "singleton" in info_tuple[1] and
                str(expr[1]) > str(expr[2])):
            return ["filter"]

        # Forbid expressions of the form (X ∩ (Y ∩ Z)),
        # because equivalent to ((X ∩ Y) ∩ Z).
        if expr[2][0] == "union":
            return ["filter"]

        # Forbid expressions of the form (X ∪ Y ∪ Z)
        # where one is a subset of A and another is a superset of A
        # (and similarly for B).
        if ("subsetOfA" in info_tuple[0] and
                expr[2][0] == "union" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("subsetOfA" in info_tuple[0] and
                expr[2][0] == "union" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if ("supersetOfA" in info_tuple[0] and
                expr[2][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("supersetOfA" in info_tuple[0] and
                expr[2][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "subsetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "subsetOfA" in info_tuple[1]):
            return ["filter"]
        #
        if ("subsetOfB" in info_tuple[0] and
                expr[2][0] == "union" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("subsetOfB" in info_tuple[0] and
                expr[2][0] == "union" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if ("supersetOfB" in info_tuple[0] and
                expr[2][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("supersetOfB" in info_tuple[0] and
                expr[2][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "subsetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "union" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "subsetOfB" in info_tuple[1]):
            return ["filter"]

        # Forbid expressions of the form X ∪ (Y \ Z) where X is a
        # superset of A and Z is a subset of A.
        if (expr[2][0] == "diff" and
                "supersetOfA" in info_tuple[0] and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "supersetOfA" in info_tuple[1] and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]
        if (expr[2][0] == "diff" and
                "supersetOfB" in info_tuple[0] and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "supersetOfB" in info_tuple[1] and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]

        return info

    # For expressions formed with the intersection operator:
    if main_operator == "intersection":
        # Label them with "combinedset" if they are not filtered out
        info = ["combinedset"]

        # Add the labels "subsetOfA"/"supersetOfA"/etc as needed
        if "subsetOfA" in info_tuple[0]:
            info.append("subsetOfA")
        if "subsetOfA" in info_tuple[1]:
            info.append("subsetOfA")
        if "subsetOfB" in info_tuple[0]:
            info.append("subsetOfB")
        if "subsetOfB" in info_tuple[1]:
            info.append("subsetOfB")
        if "supersetOfA" in info_tuple[0] and "supersetOfA" in info_tuple[1]:
            info.append("supersetOfA")
        if "supersetOfB" in info_tuple[0] and "supersetOfB" in info_tuple[1]:
            info.append("supersetOfB")
        if "atmostsingleton" in info_tuple[0]:
            info.append("atmostsingleton")
        if "atmostsingleton" in info_tuple[1]:
            info.append("atmostsingleton")

        # Intersecting a subset of A with a superset of A will just give the
        # subset, so we can filter these expressions (similarly for B)
        if "supersetOfA" in info_tuple[0] and "subsetOfA" in info_tuple[1]:
            return ["filter"]
        if "supersetOfA" in info_tuple[1] and "subsetOfA" in info_tuple[0]:
            return ["filter"]
        if "supersetOfB" in info_tuple[0] and "subsetOfB" in info_tuple[1]:
            return ["filter"]
        if "supersetOfB" in info_tuple[1] and "subsetOfB" in info_tuple[0]:
            return ["filter"]

        # We only allow primary sets on the left hand side.
        if "primary" in info_tuple[1]:
            return ["filter"]

        # For intersecting combinedsets with singletons, only allow one
        # order (where the singleton is on the right hand side)
        if ("singleton" in info_tuple[0] and
                "combinedset" in info_tuple[1]):
            return ["filter"]

        # For intersecting combinedsets with combinedsets or
        # or intersecting singletons with singletons, only allow them
        # in alphabetic order (to remove duplicates).
        if ("combinedset" in info_tuple[0] and
                "combinedset" in info_tuple[1] and
                str(expr[1]) > str(expr[2])):
            return ["filter"]
        if ("singleton" in info_tuple[0] and
                "singleton" in info_tuple[1] and
                str(expr[1]) > str(expr[2])):
            return ["filter"]

        # Forbid expressions of the form (X ∩ (Y ∩ Z)),
        # because equivalent to ((X ∩ Y) ∩ Z).
        if expr[2][0] == "intersection":
            return ["filter"]

        # Forbid expressions of the form (X ∩ Y ∩ Z)
        # where one is a subset of A and another is a superset of A
        # (and similarly for B).
        if ("subsetOfA" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("subsetOfA" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if ("supersetOfA" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("supersetOfA" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "subsetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "subsetOfA" in info_tuple[1]):
            return ["filter"]
        #
        if ("subsetOfB" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("subsetOfB" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if ("supersetOfB" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][1])):
            return ["filter"]
        if ("supersetOfB" in info_tuple[0] and
                expr[2][0] == "intersection" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[1][1]) and
                "subsetOfB" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "intersection" and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "subsetOfB" in info_tuple[1]):
            return ["filter"]

        # Forbid expressions of the form i(c,X) ∩ i(c',X) for
        # constants c != c'.
        if (expr[1][0] == "index" and
                str(expr[1][1]).isdigit() and
                expr[2][0] == "index" and
                str(expr[2][1]).isdigit() and
                str(expr[1][1]) != str(expr[2][1]) and
                expr[1][2] == expr[2][2]):
            return ["filter"]

        # Forbid expressions of the form X ∩ (Y \ Z) where X is a
        # subset of A and Z is a superset of A.
        if (expr[2][0] == "diff" and
                "subsetOfA" in info_tuple[0] and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "subsetOfA" in info_tuple[1] and
                "supersetOfA" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]
        if (expr[2][0] == "diff" and
                "subsetOfB" in info_tuple[0] and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[2][2])):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "subsetOfB" in info_tuple[1] and
                "supersetOfB" in pre_filter_recursive(lang_gen, expr[1][2])):
            return ["filter"]

        return info

    # For expressions formed with the difference operator:
    if main_operator == "diff":
        # Label them with "combinedset" if they are not filtered out
        info = ["combinedset"]

        # Add the labels "subsetOfA"/"atmostsingleton"/etc as needed
        if "subsetOfA" in info_tuple[0]:
            info.append("subsetOfA")
        if "subsetOfB" in info_tuple[0]:
            info.append("subsetOfB")
        if "atmostsingleton" in info_tuple[0]:
            info.append("atmostsingleton")

        # A subset of A minus a superset of A is always empty, so we
        # can filter it out (similarly for B)
        if ("subsetOfA" in info_tuple[0] and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if ("subsetOfB" in info_tuple[0] and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]

        # Forbid subtracting a primary set from another primary set
        # (because it is another primary set, which we already have)
        if ("primary" in info_tuple[0] and
                "primary" in info_tuple[1]):
            return ["filter"]

        # Filter out expressions of the form (X \ Y) \ Z where
        # Y is a subset of A and Z is a superset of A, because
        # equivalent to (X \ Z), and similarly for B.
        if (expr[1][0] == "diff" and
                "subsetOfA" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfA" in info_tuple[1]):
            return ["filter"]
        if (expr[1][0] == "diff" and
                "subsetOfB" in pre_filter_recursive(lang_gen, expr[1][2]) and
                "supersetOfB" in info_tuple[1]):
            return ["filter"]

        # Forbid expressions of the form i(c,X) \ i(c',X) for
        # constants c != c'.
        if (expr[1][0] == "index" and
                str(expr[1][1]).isdigit() and
                expr[2][0] == "index" and
                str(expr[2][1]).isdigit() and
                str(expr[1][1]) != str(expr[2][1]) and
                expr[1][2] == expr[2][2]):
            return ["filter"]

        # Forbid expressions of the form (X \ Y) \ X, because
        # equivalent to the empty set.
        if (expr[1][0] == "diff" and expr[1][1] == expr[2]):
            return ["filter"]

        # Forbid expressions of the form X \ (Y \ X), because
        # equivalent to X.
        if (expr[2][0] == "diff" and expr[2][2] == expr[1]):
            return ["filter"]

        # Forbid expressions of the form X \ (X \ Y), because
        # equivalent to X ∩ Y.
        if "intersection" in lang_gen.operators:
            if (expr[2][0] == "diff" and expr[2][1] == expr[1]):
                return ["filter"]

        return info

    # Filter out expressions of the form index(0, set), because it
    # represents the empty set, which is never part of any shortest
    # expression with 'unique meaning'.
    if main_operator == "index":
        info = ["singleton", "atmostsingleton"]

        # Forbid taking the 0'th element of something (always results
        # in the empty set)
        if str(expr[1]) == "0":
            return ["filter"]

        # Forbid taking the first element of something that is at
        # most a singleton (because it will just give the same set again).
        if ("atmostsingleton" in info_tuple[1] and expr[1] == "1"):
            return ["filter"]

        # Forbid taking the c'th element of a singleton for any
        # constant c, because it's either the empty set or the
        # singleton itself.
        if (str(expr[1]).isdigit() and
                expr[2][0] == "index"):
            return ["filter"]

        # Add the labels "subsetOfA" and "subsetOfB" as needed
        if "subsetOfA" in info_tuple[1]:
            info.append("subsetOfA")
        if "subsetOfB" in info_tuple[1]:
            info.append("subsetOfB")

        return info

    # If none of the cases applies, return the empty list.
    return []

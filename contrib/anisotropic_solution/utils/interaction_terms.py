from __future__ import absolute_import
import numpy as np
import re
from sympy import symbols, expand, Add


def transform_terms(power, inds, prefactor):
    s = symbols("a, b, c")
    a, b, c = s

    # premultiplying by abc makes processing easier later on
    expr = expand(
        a
        * b
        * c
        * ((s[inds[0]] + s[inds[1]] - s[inds[2]]) ** power - (a + b + c) ** power)
    )

    terms = [
        [x for x in re.split("\*", str(term)) if x] for term in Add.make_args(expr)
    ]

    new_terms = []
    for term in terms:
        coeff = np.array([float(term[0])]) * prefactor
        ap = term[term.index("a") + 1]
        bp = term[term.index("b") + 1]

        if ap == "b":
            ap = 1
        if bp == "c":
            bp = 1
        if term.index("c") + 1 == len(term):
            cp = 1
        else:
            cp = term[term.index("c") + 1]

        new_terms.append(
            [
                *coeff,
                *[0 for i in range(int(ap) - 1)],
                *[1 for i in range(int(bp) - 1)],
                *[2 for i in range(int(cp) - 1)],
            ]
        )

    return new_terms

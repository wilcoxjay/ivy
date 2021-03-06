#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
"""
This file currently defines the following classes:
* Concept
* ConceptCombiner
* ConceptDomain
"""

from itertools import product, chain
from collections import OrderedDict, defaultdict

from general import IvyError
from utils.recstruct_object import record
from logic import (Var, Const, Apply, Eq, Not, And, Or, Implies,
                   ForAll, Exists)
from logic import UninterpretedSort, FunctionSort, first_order_sort, Boolean, TopSort, SortError, contains_topsort
from logic_util import free_variables, substitute, substitute_apply, used_constants
from type_inference import concretize_sorts
from z3_utils import z3_implies


class ArityError(IvyError):
    pass


class Concept(record('Concept', ['variables', 'formula'])):
    """
    A concept is a quantifier-free formula with some first-order free variables
    """
    @classmethod
    def _preprocess_(cls, variables, formula):
        variables = tuple(variables)
        if set(variables) != free_variables(formula):
            raise IvyError("Free variables {} must match formula: {}".format(variables, formula))
        if not all(type(v) is Var and first_order_sort(v.sort) for v in variables):
            raise IvyError("Concept variables must be first-order: {}".format(variables))
        return variables, formula

    arity = property(lambda self: len(self.variables))
    sorts = property(lambda self: tuple(v.sort for v in self.variables))

    def __call__(self, *terms):
        if len(terms) != self.arity:
            raise ArityError(self.arity, terms)
        return substitute(self.formula, dict(
            (v, concretize_sorts(t, v.sort))
            for v, t in zip(self.variables, terms)
        ))

    def __str__(self):
        return 'Concept {}. {}'.format(
            ', '.join(str(v) for v in self.variables),
            self.formula,
        )


class ConceptCombiner(record('Concept', ['variables', 'formula'])):
    """
    A concept combiner is a formula with some second-order free
    variables that represent relations (i.e. have Boolean range).

    The quantifier structure of a concept combiner should be AE for
    the checks to stay in EPR.
    """
    @classmethod
    def _preprocess_(cls, variables, formula):
        variables = tuple(variables)
        if set(variables) != free_variables(formula):
            raise IvyError("Free variables {} must match formula: {}".format(variables, formula))
        if not all(type(v) is Var and
                   type(v.sort) is FunctionSort and
                   v.sort.range == Boolean
                   for v in variables):
            raise IvyError("ConceptCombiner variables must be relational: {}".format(variables))
        return variables, formula

    arity = property(lambda self: len(self.variables))
    arities = property(lambda self: tuple(len(v.sort.domain) for v in self.variables))

    def __call__(self, *concepts):
        if len(concepts) != self.arity:
            raise ArityError(self.arity, concepts)
        assert all(type(c) is Concept for c in concepts)
        if tuple(c.arity for c in concepts) != self.arities:
            raise ArityError(self.arities, concepts)
        return concretize_sorts(substitute_apply(
            self.formula,
            zip(self.variables, concepts),
            by_name=True,
        ))

    def __str__(self):
        return 'ConceptCombiner {}. {}'.format(
            ', '.join(str(v) for v in self.variables),
            self.formula,
        )


def _union_lists(lists):
    """
    Return a list that contains of all the elements of lists without
    duplicates and maintaining the order.
    """
    seen = {}
    return [seen.setdefault(x, x)
            for l in lists for x in l
            if x not in seen]


def _resolve_name(d, x):
    """
    d is a mapping, and x is a key in d or an iterable of
    those. Returns a list of all names of atoms in d that x selects.

    Atoms are anything except lists, tuples, and sets.

    If x is '*', all atom names in d are returned.

    Names missing from d are ignored and will not appear in results.

    The result is a list with no duplicates.

    If d contains cyclic references this will go into an infinite loop.
    """
    if x == '*':
        return _resolve_name(d, [k for k in d])
    elif isinstance(x, basestring):
        v = d.get(x, ())
        if type(v) in (tuple, list, set):
            return _resolve_name(d, v)
        else:
            return [x]
    else:
        return _union_lists(_resolve_name(d, k) for k in x)


def _replace_name(x, name, new):
    """
    Return a list obtained from x by replacing occurances of name with
    the list new and flattening.
    """
    if isinstance(x, basestring):
        return x
    else:
        new = list(new)
        return list(chain(*(new if y == name else [y] for y in x)))


class ConceptDomain(object):
    """
    A concept domain is a set of named concepts, named concept
    combiners, and a list of combinations.

    Each name can resolve to a list of names representing a set.

    A combination is a tuple where the first element is a combination
    label, the second element is a set of names of concept combiners
    (with the same arities), and the rest of the elements are sets of
    names of concepts such that the cartesian product of the tuple
    defines a valid set of concept instantiations.

    A ConceptDomain is mutable, so if needed it should be .copy()'ed
    before changing it.

    The special name '*' refers to all applicable objects.

    The concept domain will try to instantiate all combinations, and
    ill-sorted combinations are silently ignored.
    """
    def __init__(self, concepts, combiners, combinations):
        self.concepts = OrderedDict(concepts)
        self.combiners = OrderedDict(combiners)
        self.combinations = list(combinations)

    def __copy__(self):
        # note that the constructor does another copy
        return type(self)(self.concepts, self.combiners, self.combinations)

    def copy(self):
        return self.__copy__()

    def concepts_by_arity(self, a):
        return [name for name, concept in self.concepts.iteritems()
                if type(concept) is Concept and concept.arity == a]

    def possible_node_labes(self):
        nodes = set(self.concepts['nodes'])
        return [
            name for name, concept in self.concepts.iteritems() if
            type(concept) is Concept and
            concept.arity == 1 and
            name not in nodes
        ]

    def get_facts(self):
        """
        Return a list of pairs of the form:
        ((combination_name, combiner_name, concept_name_1, ..., concept_name_k), formula)

        Formulas that are ill-sorted are ignored

        This function causes a combinatoric explosion that should be
        avoided by using a better type system (with high-order sort
        functors). For now I'm just using hacks to keep the
        combinatoric explosion under control.
        """
        facts = []
        for combination in self.combinations:
            combination_name = combination[0]
            combiner_names = _resolve_name(self.combiners, combination[1])
            concept_names = [_resolve_name(self.concepts, x) for x in combination[2:]]
            # this combinatoric loop should be avoided using sorts...
            for concept_combo in product(*concept_names):
                concepts = [self.concepts[n] for n in concept_combo]
                for combiner_name in combiner_names:
                    combiner = self.combiners[combiner_name]
                    try:
                        formula = combiner(*concepts)
                        # assertion removed for speedup...
                        # assert not contains_topsort(formula)
                    except (SortError, ArityError):
                        # silently ignore this instantiation, assuming
                        # all other combiners in this combination are
                        # also not suitable for it
                        break
                    else:
                        tag = (combination_name, combiner_name) + concept_combo
                        facts.append((tag, formula))
        return facts

    def split(self, concept, split_by):
        """
        concept and split_by are concept names.

        This splits concept into 2 concepts:
        (concept+split_by) and (concept-split_by)
        """
        c1 = self.concepts[concept]
        c2 = self.concepts[split_by]
        variables = c1.variables
        formula_plus = And(c1(*variables), c2(*variables))
        formula_minus = And(c1(*variables), Not(c2(*variables)))
        new_names = '({}+{})'.format(concept, split_by), '({}-{})'.format(concept, split_by)
        new_concepts = Concept(variables, formula_plus), Concept(variables, formula_minus)
        names = self.concepts.keys()
        self.concepts.update(zip(new_names, new_concepts))
        self.concepts = OrderedDict((k, self.concepts[k]) for k in _replace_name(names, concept, new_names))
        self.replace_concept(concept, new_names)

    def replace_concept(self, concept, new):
        """
        concept is a concept name, and new is a list of names to replace it with
        """
        for k, v in self.concepts.iteritems():
            if type(v) is not Concept:
                self.concepts[k] = _replace_name(v, concept, new)
        self.combinations = [
            combination[:1] + tuple(_replace_name(x, concept, new) for x in combination[1:])
            for combination in self.combinations
        ]

    def output(self):
        print "Concepts:"
        for k, v in self.concepts.iteritems():
            print '   ', k, ':', v
        print "Concept Combiners:"
        for k, v in self.combiners.iteritems():
            print '   ', k, ':', v
        print "Combinations:"
        for v in self.combinations:
            print '   ', v


def get_standard_combiners():
    T = TopSort()
    UnaryRelation = FunctionSort(T, Boolean)
    BinaryRelation = FunctionSort(T, T, Boolean)
    X, Y, Z = (Var(n, T) for n in ['X', 'Y', 'Z'])
    U = Var('U', UnaryRelation)
    U1 = Var('U1', UnaryRelation)
    U2 = Var('U2', UnaryRelation)
    B = Var('B', BinaryRelation)
    B1 = Var('B1', BinaryRelation)
    B2 = Var('B2', BinaryRelation)
    result = OrderedDict()

    result['none'] = ConceptCombiner([U], Not(Exists([X], U(X))))
    result['at_least_one'] = ConceptCombiner([U], Exists([X], U(X)))
    result['at_most_one'] = ConceptCombiner([U], ForAll([X,Y], Implies(And(U(X), U(Y)), Eq(X,Y))))

    result['node_necessarily'] = ConceptCombiner(
        [U1, U2],
        ForAll([X], Implies(U1(X), U2(X))),
    )
    result['node_necessarily_not'] = ConceptCombiner(
        [U1, U2],
        ForAll([X], Implies(U1(X), Not(U2(X)))),
    )


    result['mutually_exclusive'] = ConceptCombiner(
        [U1, U2],
        ForAll([X, Y], Not(And(U1(X), U2(Y))))
    )

    result['all_to_all'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([X,Y], Implies(And(U1(X), U2(Y)), B(X,Y)))
    )
    result['none_to_none'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([X,Y], Implies(And(U1(X), U2(Y)), Not(B(X,Y))))
    )
    result['total'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([X], Implies(U1(X), Exists([Y], And(U2(Y), B(X,Y)))))
    )
    result['functional'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([X, Y, Z], Implies(And(U1(X), U2(Y), U2(Z), B(X,Y), B(X,Z)), Eq(Y,Z)))
    )
    result['surjective'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([Y], Implies(U2(Y), Exists([X], And(U1(X), B(X,Y)))))
    )
    result['injective'] = ConceptCombiner(
        [B, U1, U2],
        ForAll([X, Y, Z], Implies(And(U1(X), U1(Y), U2(Z), B(X,Z), B(Y,Z)), Eq(X,Y)))
    )

    result['node_info'] = ['none', 'at_least_one', 'at_most_one']
    if False:
        # this just slows us down, and it's not clear it's needed
        # later this should be made customizable by the user
        result['edge_info'] = ['all_to_all', 'none_to_none', 'total',
                               'functional', 'surjective', 'injective']
    else:
        result['edge_info'] = ['all_to_all', 'none_to_none']

    result['node_label'] = ['node_necessarily', 'node_necessarily_not']

    return result


def get_standard_combinations():
    return [
        ('node_info', 'node_info', 'nodes'),
        ('edge_info', 'edge_info', 'edges', 'nodes', 'nodes'),
        ('node_label', 'node_label', 'nodes', 'node_labels'),
    ]


def get_initial_concept_domain(sig):
    """
    sig is an ivy_logic.Sig object
    """
    concepts = OrderedDict()

    concepts['nodes'] = []
    concepts['node_labels'] = []
    concepts['edges'] = []

    # add sort concepts
    for s in sorted(sig.sorts.values()):
        X = Var('X', s)
        concepts[s.name] = Concept([X], Eq(X,X))
        concepts['nodes'].append(s.name)

    # add equality concept
    X = Var('X', TopSort())
    Y = Var('Y', TopSort())
    concepts['='] = Concept([X, Y], Eq(X, Y))

    # add concepts from relations
    for c in sorted(sig.symbols.values()):
        assert type(c) is Const

        if first_order_sort(c.sort):
            # first order constant, add unary equality concept
            X = Var('X', c.sort)
            name = '={}'.format(c.name)
            concepts[name] = Concept([X], Eq(X,c))

        elif type(c.sort) is FunctionSort and c.sort.arity == 1:
            # add unary concept and label
            X = Var('X', c.sort.domain[0])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X], c(X))

        elif type(c.sort) is FunctionSort and c.sort.arity == 2:
            # add binary concept and edge
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y], c(X, Y))

        elif type(c.sort) is FunctionSort and c.sort.arity == 3:
            # add ternary concept
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            Z = Var('Z', c.sort.domain[2])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y, Z], c(X, Y, Z))

        else:
            # skip other symbols
            pass

    return ConceptDomain(concepts, get_standard_combiners(), get_standard_combinations())


def get_diagram_concept_domain(sig, diagram):
    """
    sig is an ivy_logic.Sig object
    diagram is a formula
    """
    concepts = OrderedDict()

    concepts['nodes'] = []
    concepts['node_labels'] = []
    concepts['edges'] = []

    # add equality concept
    X = Var('X', TopSort())
    Y = Var('Y', TopSort())
    concepts['='] = Concept([X, Y], Eq(X, Y))

    # add concepts from relations and constants in the signature and
    # in the diagram
    if sig is not None:
        sig_symbols = frozenset(sig.symbols.values())
    else:
        sig_symbols = frozenset()
    for c in sorted(sig_symbols | used_constants(diagram)):
        assert type(c) is Const

        if first_order_sort(c.sort):
            # first order constant, add unary equality concept
            X = Var('X', c.sort)
            name = '{}:{}'.format(c.name, c.sort)
            concepts[name] = Concept([X], Eq(X,c))
            concepts['nodes'].append(name)

        elif type(c.sort) is FunctionSort and c.sort.arity == 1:
            # add unary concept and label
            X = Var('X', c.sort.domain[0])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X], c(X))

        elif type(c.sort) is FunctionSort and c.sort.arity == 2:
            # add binary concept and edge
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y], c(X, Y))

        elif type(c.sort) is FunctionSort and c.sort.arity == 3:
            # add ternary concept
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            Z = Var('Z', c.sort.domain[2])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y, Z], c(X, Y, Z))

        else:
            # skip other symbols
            pass

    return ConceptDomain(concepts, get_standard_combiners(), get_standard_combinations())


def get_structure_concept_domain(state, sig=None):
    """
    state is an ivy_interp.State with a .universe
    sig is an ivy_logic.Sig object
    """
    concepts = OrderedDict()

    concepts['nodes'] = []
    concepts['node_labels'] = []

    # add equality concept
    X = Var('X', TopSort())
    Y = Var('Y', TopSort())
    concepts['='] = Concept([X, Y], Eq(X, Y))

    # add nodes for universe elements
    elements = [uc for s in state.universe for uc in state.universe[s]]
    for uc in sorted(elements):
        # add unary equality concept
        X = Var('X', uc.sort)
        name = uc.name
        if str(uc.sort) not in name:
            name += ':{}'.format(uc.sort)
        concepts[name] = Concept([X], Eq(X,uc))
        concepts['nodes'].append(name)

    # # find which symbols are equal to which universe constant
    # equals = dict(
    #     (uc, [c for c in symbols
    #           if c != uc and
    #           c.sort == s and
    #           z3_implies(state_formula, Eq(c, uc))])
    #     for s in state.universe
    #     for uc in state.universe[s]
    # )

    # add concepts for relations and constants
    state_formula = state.clauses.to_formula()
    symbols = used_constants(state_formula)
    if sig is not None:
        symbols = symbols | frozenset(sig.symbols.values())
    symbols = symbols - frozenset(elements)
    symbols = sorted(symbols)
    for c in symbols:
        assert type(c) is Const

        if first_order_sort(c.sort):
            # first order constant, add unary equality concept
            X = Var('X', c.sort)
            name = '={}'.format(c.name)
            concepts[name] = Concept([X], Eq(X,c))

        elif type(c.sort) is FunctionSort and c.sort.arity == 1:
            # add unary concept and label
            X = Var('X', c.sort.domain[0])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X], c(X))

        elif type(c.sort) is FunctionSort and c.sort.arity == 2:
            # add binary concept and edge
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y], c(X, Y))

        elif type(c.sort) is FunctionSort and c.sort.arity == 3:
            # add ternary concept
            X = Var('X', c.sort.domain[0])
            Y = Var('Y', c.sort.domain[1])
            Z = Var('Z', c.sort.domain[2])
            name = '{}'.format(c.name)
            concepts[name] = Concept([X, Y, Z], c(X, Y, Z))

        else:
            # skip other symbols
            pass

    return ConceptDomain(concepts, get_standard_combiners(), get_standard_combinations())


def get_structure_concept_abstract_value(state):
    """
    state is an ivy_interp.State with a .universe

    result can be used as a cache for concept_alpha.alpha
    """
    abstract_value = []

    # add node_info results for universe elements
    elements = [uc for s in state.universe for uc in state.universe[s]]
    nodes = []
    for uc in sorted(elements):
        name = uc.name
        nodes.append(name)
        abstract_value += [
            (('node_info', 'none', name), False),
            (('node_info', 'at_least_one', name), True),
            (('node_info', 'at_most_one', name), True),
        ]
    nodes = frozenset(nodes)

    # add concepts for relations and constants
    state_formula = state.clauses.to_formula()
    assert type(state_formula) is And
    for lit in state_formula:
        if type(lit) is ForAll:
            # universe constaint
            continue

        if type(lit) is Not:
            polarity = False
            lit = lit.body
        else:
            polarity = True

        if type(lit) is Eq:
            if lit.t1.name not in nodes:
                assert polarity is True
                label_name = '={}'.format(lit.t1.name)
                for node_name in nodes:
                    polarity = node_name == lit.t2.name
                    abstract_value += [
                        (('node_label', 'node_necessarily', node_name, label_name), polarity),
                        (('node_label', 'node_necessarily_not', node_name, label_name), not polarity),
                    ]

        elif type(lit) is Apply and lit.func.sort.arity == 1:
            label_name = lit.func.name
            node_name = lit.terms[0].name
            abstract_value += [
                (('node_label', 'node_necessarily', node_name, label_name), polarity),
                (('node_label', 'node_necessarily_not', node_name, label_name), not polarity),
            ]

        elif type(lit) is Apply and lit.func.sort.arity == 2:
            edge_name = lit.func.name
            source_name = lit.terms[0].name
            target_name = lit.terms[1].name
            abstract_value += [
                (('edge_info', 'all_to_all', edge_name, source_name, target_name),
                 polarity),
                (('edge_info', 'none_to_none', edge_name, source_name, target_name),
                 not polarity),
            ]

        elif type(lit) is Apply and lit.func.sort.arity > 2:
            pass # TODO: can also support projections here

        else:
            pass
#            assert False, lit

    return dict(abstract_value)


def get_structure_renaming(state):
    """
    state is an ivy_interp.State with a .universe

    result is a dictionary mapping universe constant names to prettier
    names that should be used for displaying to the user
    """
    from ivy_utils import topological_sort

    # add node_info results for universe elements
    elements = list(set([uc for s in state.universe for uc in state.universe[s]]))

    # name according to topological sort
    # TODO: make this not specific to leader example
    state_formula = state.clauses.to_formula()
    assert type(state_formula) is And
    order = []
    for lit in state_formula:
        if type(lit) is Apply and lit.func.sort.arity == 2 and lit.func.name in ('reach', 'le'):
            order.append(lit.terms)
    elements = topological_sort(elements, order, lambda c: c.name)

    result = dict()
    count = defaultdict(int)
    for c in elements:
        prefix = str(c.sort).lower()
        result[c.name] = prefix + str(count[prefix])
        count[prefix] += 1

    return result


if __name__ == '__main__':
    def test(st):
        print st, "=", eval(st)

    S = UninterpretedSort('S')
    T = TopSort()
    UnaryRelationS = FunctionSort(S, Boolean)
    BinaryRelationS = FunctionSort(S, S, Boolean)
    UnaryRelationT = FunctionSort(T, Boolean)
    BinaryRelationT = FunctionSort(T, T, Boolean)


    X, Y, Z = (Var(n, S) for n in ['X', 'Y', 'Z'])

    r = Const('r', BinaryRelationS)
    n = Const('n', BinaryRelationS)
    p = Const('p', UnaryRelationS)
    q = Const('q', UnaryRelationS)
    u = Const('u', UnaryRelationS)

    c11 = Concept([X], And(p(X), q(X)))
    c10 = Concept([X], And(p(X), Not(q(X))))
    c01 = Concept([X], And(Not(p(X)), q(X)))
    c00 = Concept([X], And(Not(p(X)), Not(q(X))))
    cu = Concept([X], u(X))

    crn = Concept([X, Y], Or(r(X,Y), n(X,Y)))

    combiners = get_standard_combiners()
    combinations = get_standard_combinations()

    test('c11')
    test('c10')
    test('c01')
    test('c00')
    test('cu')
    test('crn')
    print
    for k, v in combiners.iteritems():
        print k, ':', v
    print

    cd = ConceptDomain(
        OrderedDict([
            ('both', c11),
            ('none', c00),
            ('onlyp', c10),
            ('onlyq', c01),
            ('r_or_n', crn),
            ('u', cu),
            ('nodes', ['both', 'none', 'onlyp', 'onlyq']),
            ('edges', ['r_or_n']),
            ('node_labels', ['u']),
        ]),
        combiners,
        combinations,
    )

    print "cd = ("
    cd.output()
    print ")\n"

    facts = cd.get_facts()
    if True:
        print "facts ({}) = [".format(len(facts))
        for tag, formula in facts:
            print '   ', tag, ':', formula, ','
        print "]\n"

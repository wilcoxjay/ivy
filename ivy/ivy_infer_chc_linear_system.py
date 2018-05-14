#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
import ivy_utils as utl
import ivy_utils as iu
import ivy_module as im
import ivy_art
import ivy_interp
import ivy_isolate

import ivy_logic_utils

import sys
import logging

import ivy_init

from ivy_infer import ClausesClauses
import ivy_infer
import ivy_linear_pdr
import ivy_interp as itp
import ivy_infer_universal
import ivy_transrel
import ivy_actions
import ivy_solver

logger = logging.getLogger(__file__)

# TODO: eliminate duplication with ivy_infer_global_invariant
def global_initial_state():
    # see ivy_check.summarize_isolate()
    with im.module.copy():
        with itp.EvalContext(check=False):
            ag = ivy_art.AnalysisGraph(initializer=lambda x: None)
            assert len(ag.states) == 1
            history = ag.get_history(ag.states[0])
            initial_state_clauses = history.post
            logger.debug("initial state clauses: %s", initial_state_clauses)
            return initial_state_clauses

def ivy_all_axioms():
    axioms_lst = [ivy_logic_utils.formula_to_clauses(lc.formula) for lc in im.module.labeled_axioms]
    if axioms_lst:
        return ivy_logic_utils.and_clauses(*axioms_lst)
    # and_clauses on an empty list causes problems, later fails in clauses_using_symbols
    return ivy_logic_utils.true_clauses()


def get_domain():
    return ivy_art.AnalysisGraph().domain

def action_update(action):
    pre = ivy_interp.State()
    return action.update(get_domain(), pre.in_scope)


def global_initial_state():
    # see ivy_check.summarize_isolate()
    with im.module.copy():
        with itp.EvalContext(check=False):
            ag = ivy_art.AnalysisGraph(initializer=lambda x: None)
            assert len(ag.states) == 1
            history = ag.get_history(ag.states[0])
            initial_state_clauses = history.post
            logger.debug("initial state clauses: %s", initial_state_clauses)
            return initial_state_clauses


def ivy_all_axioms():
    axioms_lst = [ivy_logic_utils.formula_to_clauses(lc.formula) for lc in im.module.labeled_axioms]
    if axioms_lst:
        return ivy_logic_utils.and_clauses(*axioms_lst)
    # and_clauses on an empty list causes problems, later fails in clauses_using_symbols
    return ivy_logic_utils.true_clauses()


def check_action_transition(prestate_clauses, action_name, poststate_obligation):
    import ivy_logic as il
    import logic as lg
    from ivy_interp import EvalContext
    import ivy_module as im
    import ivy_logic_utils as ilu
    from ivy_solver import get_small_model
    from ivy_logic_utils import and_clauses, dual_clauses
    from ivy_interp import State

    if True:
        # relying on isolate context created earlier
        ag = ivy_art.AnalysisGraph()

        pre = State()
        pre.clauses = and_clauses(*prestate_clauses.get_conjuncts_clauses_list())
        pre.clauses = and_clauses(pre.clauses, ivy_all_axioms())

        with EvalContext(check=False):
            post = ag.execute_action(action_name, pre, None)

        post.clauses = ilu.true_clauses()

        to_test = poststate_obligation.get_conjuncts_clauses_list()

        while len(to_test) > 0:
            conj = to_test.pop(0)
            used_names = frozenset(x.name for x in il.sig.symbols.values())

            def witness(v):
                c = lg.Const('@' + v.name, v.sort)
                assert c.name not in used_names
                return c

            clauses = dual_clauses(conj, witness)
            history = ag.get_history(post)

            _get_model_clauses = lambda clauses, final_cond=False: get_small_model(
                clauses,
                sorted(il.sig.sorts.values()),
                [],
                final_cond=final_cond
            )

            # res = ag.bmc(post, clauses, None, None, _get_model_clauses)
            res = ag.bmc(post, clauses)

            if res is not None:
                assert len(res.states) == 2
                return res.states
            else:
                return None

            # attempt to mimic generalize_intransformability: (29/3/2018)
            # gap is to take only the prestate of the cti and pass it forwards (to the diagram)
            #
            # ag = ivy_art.AnalysisGraph()
            #
            # pre = State()
            # pre.clauses = and_clauses(*prestate_clauses.get_conjuncts_clauses_list())
            #
            # # relying on the isolate being created with 'ext' action
            # action = im.module.actions['ext']
            #
            # post = ivy_logic_utils.dual_clauses(conj)
            #
            # axioms = ivy_all_axioms()
            # import ivy_transrel
            # pre_and_tr = ivy_transrel.forward_image(pre.clauses, axioms,
            #                                         action.update(ag.domain, pre.in_scope))
            # vc = ClausesClauses([pre_and_tr, post])
            # cti = vc.get_model()
            # if cti is None:
            #     continue
            #
            # return (vc, cti)

            # return None

        # TODO: attempt to mimic the new ivy_check (26/3/2018)
        # while len(to_test) > 0:
        #     conj = to_test.pop(0)
        #     assert conj.is_universal_first_order(), conj
        #     # used_names = frozenset(x.name for x in il.sig.symbols.values())
        #     # def witness(v):
        #     #     c = lg.Const('@' + v.name, v.sort)
        #     #     assert c.name not in used_names
        #     #     return c
        #
        #     # clauses_to_check = dual_clauses(conj, witness)
        #     clauses_to_check = dual_clauses(conj)
        #
        #     # based on ivy_check.check_fcs_in_state()
        #     history = ag.get_history(post)
        #     clauses = history.post
        #     clauses = ivy_logic_utils.and_clauses(clauses, im.module.background_theory())
        #     model = ivy_transrel.small_model_clauses(clauses, final_cond=clauses_to_check, shrink=True)
        #     if model is None:
        #         continue
        #
        #     # based on ivy_check.MatchHandler.__init__
        #     prestate_model_clauses = ivy_solver.clauses_model_to_clauses(clauses, model=model, numerals=True)
        #     return prestate_model_clauses
        #
        # return None

def global_safety_clauses_lst():
    return [ivy_logic_utils.formula_to_clauses(lc.formula) for lc in im.module.labeled_conjs]

class SafetyOfStateClause(ivy_linear_pdr.LinearSafetyConstraint):
    def __init__(self, pred, safety_clauses_lst):
        super(SafetyOfStateClause, self).__init__(pred, ivy_logic_utils.true_clauses())
        self._safey_clauses_lst = safety_clauses_lst

    def check_satisfaction(self, summaries_by_pred):
        inv_summary = summaries_by_pred[self._lhs_pred].get_summary()

        for conjecture in self._safey_clauses_lst:
            bad_clauses = ivy_logic_utils.dual_clauses(conjecture)
            inv_but_bad_clauses = ClausesClauses(inv_summary.get_conjuncts_clauses_list() + [bad_clauses])
            bad_inv_model = inv_but_bad_clauses.get_model()
            if bad_inv_model is None:
                continue

            return ivy_infer.PdrCexModel(bad_inv_model, inv_but_bad_clauses.to_single_clauses())

        return None


class AutomatonEdge(object):
    def __init__(self, action_name, precondition=ivy_logic_utils.true_clauses()):
        self._action_name = action_name
        self._precondition = precondition

    def get_action_name(self):
        return self._action_name

    def get_precondition(self):
        return self._precondition

    def __repr__(self):
        return "%s assume %s" % (self._action_name, self._precondition)


class OutEdgesCoveringTrClause(ivy_linear_pdr.LinearSafetyConstraint):
    def __init__(self, pred, out_edges_actions):
        super(OutEdgesCoveringTrClause, self).__init__(pred, ivy_logic_utils.true_clauses())

        self._out_edges_actions = out_edges_actions
        checked_wrt_to_actions = self.full_tr_list_actions()
        for out_edge in self._out_edges_actions:
            assert out_edge.get_action_name() in checked_wrt_to_actions

    def full_tr_list_actions(self):
        # excluding the action representing the disjunction of all actions
        return filter(lambda action_name: action_name != 'ext', im.module.public_actions)

    def check_satisfaction(self, summaries_by_pred):
        logging.debug("Check edge covering: all exported %s, is covered by %s", self.full_tr_list_actions(), self._out_edges_actions)

        for action_check_covered in self.full_tr_list_actions():
            matching_edges = filter(lambda edge: edge.get_action_name() == action_check_covered, self._out_edges_actions)
            # accumulated_pre = ivy_logic_utils.or_clauses(*(edge.get_precondition() for edge in matching_edges)).epr_closed()

            # check: I_s /\ TR[action] => \/ accumulated_pre
            (_, tr_action, _) = action_update(im.module.actions[action_check_covered])
            vc = ClausesClauses([summaries_by_pred[self._lhs_pred].get_summary().to_single_clauses(),
                                 tr_action] +
                                [ivy_logic_utils.dual_clauses(edge.get_precondition()) for edge in matching_edges])

            cex = vc.get_model()
            if cex is None:
                continue

            logger.debug("Check covered failed: %s doesn't cover action %s",
                         [edge.get_precondition() for edge in matching_edges],
                         action_check_covered)

            return ivy_infer.PdrCexModel(cex, vc.to_single_clauses(), project_pre=True)

        return None


class SummaryPostSummaryClause(ivy_linear_pdr.LinearMiddleConstraint):
    def __init__(self, lhs_pred, edge_action_name, rhs_pred):
        super(SummaryPostSummaryClause, self).__init__(lhs_pred, edge_action_name, rhs_pred)
        self._edge_action_name = edge_action_name

    def check_transformability(self, summaries_by_pred, bad_clauses):
        prestate_summary = summaries_by_pred[self._lhs_pred].get_summary()

        proof_obligation = ivy_logic_utils.dual_clauses(bad_clauses)

        logger.debug("Checking edge (%s, %s, %s): %s in prestate guarantees %s in poststate?",
                     self._lhs_pred, self._edge_action_name, self._rhs_pred,
                     prestate_summary.to_single_clauses(), proof_obligation)

        countertransition = check_action_transition(prestate_summary,
                                                    self._edge_action_name,
                                                    ClausesClauses([proof_obligation]))

        if countertransition is None:
            logger.debug("Proof obligation guaranteed by prestate invariant")
            return None

        prestate = countertransition[0]
        return ivy_infer.PdrCexModel(None, prestate.clauses)

    def generalize_intransformability(self, prestate_summaries, lemma):
        import ivy_transrel
        from ivy_logic_utils import and_clauses
        from ivy_interp import State

        prestate_clauses = prestate_summaries[self._lhs_pred].get_summary()

        # relying on isolate context created earlier
        ag = ivy_art.AnalysisGraph()

        pre = State()
        pre.clauses = and_clauses(*prestate_clauses.get_conjuncts_clauses_list())

        action = im.module.actions[self._edge_action_name]

        post = ivy_logic_utils.dual_clauses(lemma)

        axioms = ivy_all_axioms()
        NO_INTERPRETED = None
        res = ivy_transrel.forward_interpolant(pre.clauses, action.update(ag.domain, pre.in_scope), post, axioms,
                                               NO_INTERPRETED)
        assert res != None
        return res[1]


def out_edge_covering_tr_constraints(states, edges):
    constraints = []
    for state in states:
        out_actions = [AutomatonEdge(action, precondition=pre) for (s1, _, action, pre) in edges if s1 == state]
        constraints.append(OutEdgesCoveringTrClause(state, out_actions))

    return constraints

def parse_json_automaton(filename):
    import json
    with open(filename, 'rt') as f:
        file_contents = f.read()
    json_data = json.loads(file_contents)

    states = [s['name'] for s in json_data['states']]
    init = [(json_data['init'], global_initial_state())]
    edges = []
    for s in json_data['states']:
        for e in s['edges']:
            target = e['target']
            action = e['action']
            if 'precondition' in e:
                precondition = ivy_logic_utils.to_clauses(e['precondition'])
            else:
                precondition = ivy_logic_utils.true_clauses()
            edges.append((s['name'], target, action, precondition))
    safety_str = json_data['safety']
    if not safety_str:
        safety = global_safety_clauses_lst()
    else:
        safety = [ivy_logic_utils.to_clauses(safety_str)]

    return states, init, edges, safety

def infer_safe_summaries(automaton_filename):
    states, init, edges, safety_clauses_lst = parse_json_automaton(automaton_filename)
    logger.debug("States: %s", states)
    logger.debug("Init: %s", init)
    logger.debug("Edges: %s", edges)
    logger.debug("Safety: %s", safety_clauses_lst)

    mid = [SummaryPostSummaryClause(s1, action, s2) for (s1, s2, action, _) in edges]
    end_state_safety = [SafetyOfStateClause(s, safety_clauses_lst) for s in states]
    end_state_cover_tr = out_edge_covering_tr_constraints(states, edges)
    end = end_state_safety + end_state_cover_tr

    pdr_elements_global_invariant = ivy_linear_pdr.LinearPdr(states, init, mid, end,
                                                             ivy_infer_universal.UnivGeneralizer())
    is_safe, frame_or_cex = ivy_infer.pdr(pdr_elements_global_invariant)
    if not is_safe:
        print "Possibly not safe! - bug or no universal invariant"
        cex = frame_or_cex
        while cex:
            logger.info("%s" % cex.predicate)
            assert len(cex.children) == 1
            cex = cex.children[0]
    else:
        safe_frame = frame_or_cex
        for state, summary in safe_frame.iteritems():
            logger.info("Summary of %s: %s", state, summary.get_summary())

        # TODO: algorithm for minimization?
        # invariant = safe_frame["inv"].get_summary()
        # logger.info("Invariant: %s. Time: %s", invariant, datetime.datetime.now())
        # logger.info("Invariant as a single formula: %s", invariant.to_single_clauses())
        # assert check_any_exported_action_transition(invariant, invariant) is None
        #
        # invariant_reduced_equiv = minimize_invariant(invariant.get_conjuncts_clauses_list(),
        #                                              lambda candidate, omitted: check_logical_implication(candidate,
        #                                                                                                   [omitted]))
        # assert ivy_solver.clauses_list_imply_list(invariant_reduced_equiv,
        #                                           invariant.get_conjuncts_clauses_list())
        # assert ivy_solver.clauses_list_imply_list(invariant.get_conjuncts_clauses_list(),
        #                                           invariant_reduced_equiv)
        # logger.info("Invariant reduced (logical equivalence): %s", invariant_reduced_equiv)
        #
        # invariant_reduced = minimize_invariant(invariant_reduced_equiv,
        #                                        lambda candidate_lst, omitted: check_inductive_invariant(candidate_lst))
        # print "Invariant reduced:", invariant_reduced


def usage():
    print "usage: \n  {} file.ivy".format(sys.argv[0])
    sys.exit(1)


def main():
    logging.basicConfig(level=logging.DEBUG)

    import signal
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    import ivy_alpha
    ivy_alpha.test_bottom = False  # this prevents a useless SAT check
    import tk_ui as ui
    iu.set_parameters({'mode': 'induction'})

    ivy_init.read_params()
    if len(sys.argv) != 3 or not sys.argv[1].endswith('ivy'):
        usage()
    with im.Module():
        with utl.ErrorPrinter():
            ivy_init.source_file(sys.argv[1], ivy_init.open_read(sys.argv[1]), create_isolate=False)

            # inspired by ivy_check.check_module()
            isolates = sorted(list(im.module.isolates))
            assert len(isolates) == 1
            isolate = isolates[0]
            with im.module.copy():
                ivy_isolate.create_isolate(isolate, ext='ext')
                infer_safe_summaries(sys.argv[2])

    print "OK"


if __name__ == "__main__":
    main()
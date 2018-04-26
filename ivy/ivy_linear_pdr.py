#
# Copyright (c) Microsoft Corporation. All Rights Reserved.
#
import abc

import ivy
import ivy_interp as itp
import ivy_utils as utl
import tk_ui as ui
import ivy_utils as iu
import ivy_module as im
import ivy_alpha
import ivy_art
import ivy_interp
import ivy_isolate

import ivy_logic_utils

import sys
import logging

import ivy_init
import ivy_ast

logger = logging.getLogger(__file__)

import ivy_infer
from ivy_infer import ClausesClauses
import ivy_infer_universal
import ivy_solver
import ivy_transrel

# TODO: remove from this module
def ivy_all_axioms():
    axioms_lst = [ivy_logic_utils.formula_to_clauses(lc.formula) for lc in im.module.labeled_axioms]
    if axioms_lst:
        return ivy_logic_utils.and_clauses(*axioms_lst)
    # and_clauses on an empty list causes problems, later fails in clauses_using_symbols
    return ivy_logic_utils.true_clauses()

class LinearTransformabilityHornClause(object):
    """
    Transformability clauses = Predicate on lhs.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred, lhs_constraint):
        self._lhs_pred = lhs_pred
        self._lhs_constraint = lhs_constraint

    def lhs_pred(self):
        return self._lhs_pred

    # def check_transformability(self, summaries_by_pred, bad_clauses):
        # lhs = self.lhs_assigned(summaries_by_pred)
        # rhs = self._rhs_pred.rhs_assigned(bad_clauses)
        # vc = ClausesClauses(lhs, rhs)
        # cex = vc.get_model()
        # if cex is None:
        #     return None

        # return PdrCexModel(cex)

    def lhs_assigned(self, summaries_by_pred):
        lhs_summary = summaries_by_pred[self._lhs_pred]
        substitute_lhs_summary = ivy_logic_utils.and_clauses(*lhs_summary.get_summary().get_conjuncts_clauses_list())
        return ivy_transrel.conjoin(substitute_lhs_summary, self._lhs_constraint)

class LinearSafetyConstraint(LinearTransformabilityHornClause):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred, lhs_constraint):
        super(LinearSafetyConstraint, self).__init__(lhs_pred, lhs_constraint)

    @abc.abstractmethod
    def check_satisfaction(self, summaries_by_pred):
        pass

class LinearMiddleConstraint(LinearTransformabilityHornClause):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred, lhs_constraint, rhs_pred):
        super(LinearMiddleConstraint, self).__init__(lhs_pred, lhs_constraint)
        self._rhs_pred = rhs_pred

    def rhs_pred(self):
        return self._rhs_pred

    @abc.abstractmethod
    def check_transformability(self, summaries_by_pred, bad_clauses):
        pass

class LinearPdr(ivy_infer_universal.UnivPdrElements):
    def __init__(self, preds, init_chc_lst, mid_chc_lst, end_chc_lst):
        self._preds = preds

        self._init_chc = init_chc_lst
        self._mid_chc = mid_chc_lst
        self._end_chc = end_chc_lst

    def initial_summary(self):
        initial_summary = {pred: ivy_logic_utils.false_clauses() for pred in self._preds}

        for (pred, init_requirement) in self._init_chc:
            current_init = initial_summary[pred]
            strengthened_init = ivy_logic_utils.or_clauses(init_requirement, current_init)
            initial_summary[pred] = strengthened_init

        return {pred: ivy_infer.PredicateSummary(pred, initial_summary[pred]) for pred in self._preds}

    def top_summary(self):
        return {pred: ivy_infer.PredicateSummary(pred, ivy_logic_utils.true_clauses()) for pred in self._preds}

    def push_forward(self, prev_summaries, current_summaries):
        for pred in prev_summaries:
            prev_clauses_lst = prev_summaries[pred].get_summary().get_conjuncts_clauses_list()
            current_clauses_lst = current_summaries[pred].get_summary().get_conjuncts_clauses_list()

            for clauses in prev_clauses_lst:
                if clauses in current_clauses_lst:
                    continue

                transformability_cex = self.check_transformability_to_violation(pred, prev_summaries, clauses)
                if transformability_cex:
                    continue

                logging.debug("Pushing to next frame for %s: %s", pred, clauses)
                current_summaries[pred].strengthen(clauses)

        return current_summaries

    def check_summary_safety(self, summaries):
        proof_obligations = []
        for safety_constraint in self._end_chc:
            bad_model = safety_constraint.check_satisfaction(summaries)
            if bad_model is None:
                continue

            proof_obligation = self._bad_model_to_proof_obligation(bad_model)
            proof_obligations.append((safety_constraint,
                                      [(safety_constraint.lhs_pred(), proof_obligation)]))

        return proof_obligations

    def check_transformability_to_violation(self, predicate, summaries_by_symbol, proof_obligation):
        proof_obligations = []

        for mid_constraint in self._mid_chc:
            if mid_constraint.rhs_pred() != predicate:
                continue

            bad_model_lhs = mid_constraint.check_transformability(summaries_by_symbol, ivy_logic_utils.dual_clauses(proof_obligation))
            if bad_model_lhs is None:
                continue

            proof_obligation = self._bad_model_to_proof_obligation(bad_model_lhs)
            pre_pred = mid_constraint.lhs_pred()
            proof_obligations.append((mid_constraint, [(pre_pred, proof_obligation)]))

        return proof_obligations

    def mark_reachable(self, predicate, summary_proof_obligation,
                       summaries, cex_info):
        pass

    def is_known_to_be_reachable(self, predicate, summary_proof_obligation,
                                 summaries):
        return False, None

    def generalize_intransformability(self, predicate, prestate_summaries, lemma):
        # TODO: hack for global inductive invariant, just for checking what's going on with the rest of the code :)

        import ivy_module as im
        import ivy_transrel
        from ivy_logic_utils import and_clauses
        from ivy_interp import State

        assert predicate == "inv"

        prestate_clauses = prestate_summaries["inv"].get_summary()

        # relying on isolate context created earlier
        ag = ivy_art.AnalysisGraph()

        pre = State()
        pre.clauses = and_clauses(*prestate_clauses.get_conjuncts_clauses_list())

        # relying on the isolate being created with 'ext' action
        action = im.module.actions['ext']

        post = ivy_logic_utils.dual_clauses(lemma)

        axioms = ivy_all_axioms()
        NO_INTERPRETED = None
        res = ivy_transrel.forward_interpolant(pre.clauses, action.update(ag.domain, pre.in_scope), post, axioms,
                                               NO_INTERPRETED)
        assert res != None
        return res[1]

        # transforming_constraints = filter(lambda constraint: constraint.rhs_pred() == predicate,
        #                                   self._mid_chc)
        #
        # combined_lhs_clauses = ivy_logic_utils.or_clauses(*[constraint.lhs_assigned(prestate_summaries)
        #                                                         for constraint in transforming_constraints])
        #
        # # poststate_lemma = predicate.rhs_assigned(lemma) TODO: resurrect dependency on predicate (or constraint...) on vocab of r.h.s.
        # poststate_lemma = ivy_transrel.new(lemma)
        # poststate_bad = ivy_logic_utils.dual_clauses(poststate_lemma)
        #
        # NO_EXTRA_AXIOMS = ivy_logic_utils.true_clauses() # should be part of the constraints
        # NO_INTERPRETED = None
        # res = ivy_transrel.interpolant(combined_lhs_clauses, poststate_bad, NO_EXTRA_AXIOMS, NO_INTERPRETED)
        # assert res is not None, "Failure generalizing blocked lemma: pred %s, %s" % (predicate, lemma)
        # return res[1]
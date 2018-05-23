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

# TODO: based on ivy_transrel.forward_image_map
def forward_clauses(clauses, updated):
    return ivy_transrel.rename_clauses(clauses, dict((x, ivy_transrel.new(x)) for x in updated))

def to_current_clauses(clauses, updated):
    return ivy_transrel.rename_clauses(clauses, dict((ivy_transrel.new(x), x) for x in updated))

class LinearTransformabilityHornClause(object):
    """
    Transformability clauses = Predicate on lhs.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred):
        self._lhs_pred = lhs_pred
        # self._lhs_constraint = lhs_constraint

    def lhs_pred(self):
        return self._lhs_pred

    # def lhs_assigned(self, summaries_by_pred):
    #     lhs_summary = summaries_by_pred[self._lhs_pred]
    #     substitute_lhs_summary = ivy_logic_utils.and_clauses(*lhs_summary.get_summary().get_conjuncts_clauses_list())
    #     return ivy_transrel.conjoin(substitute_lhs_summary, self._lhs_constraint)

class LinearSafetyConstraint(LinearTransformabilityHornClause):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred, lhs_constraint):
        super(LinearSafetyConstraint, self).__init__(lhs_pred)

    @abc.abstractmethod
    def check_satisfaction(self, summaries_by_pred):
        pass

class LinearMiddleConstraint(LinearTransformabilityHornClause):
    __metaclass__ = abc.ABCMeta

    def __init__(self, lhs_pred, lhs_constraint, rhs_pred):
        super(LinearMiddleConstraint, self).__init__(lhs_pred)
        self._rhs_pred = rhs_pred
        self._lhs_constraint = lhs_constraint

    def rhs_pred(self):
        return self._rhs_pred

    @abc.abstractmethod
    def check_transformability(self, summaries_by_pred, bad_clauses):
        pass

    @abc.abstractmethod
    def generalize_intransformability(self, summaries_by_pred, lemma):
        pass

    @abc.abstractmethod
    def transformability_update(self, summaries_by_pred, rhs_vocab):
        pass

class LinearPdr(ivy_infer.PdrElements):
    def __init__(self, preds, init_chc_lst, mid_chc_lst, end_chc_lst, generalizer, axioms):
        super(LinearPdr, self).__init__(generalizer)
        self._preds = preds

        self._init_chc = init_chc_lst
        self._mid_chc = mid_chc_lst
        self._end_chc = end_chc_lst

        self._axioms = axioms

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

            proof_obligation = self._generalizer.bad_model_to_proof_obligation(bad_model)
            proof_obligations.append((safety_constraint,
                                      [(safety_constraint.lhs_pred(), proof_obligation)]))

        return proof_obligations

    def check_transformability_to_violation(self, predicate, summaries_by_symbol, proof_obligation):
        proof_obligations = []

        for mid_constraint in self._mid_chc:
            if mid_constraint.rhs_pred() != predicate:
                continue

            logging.debug("Proof obligation: %s", proof_obligation)
            bad_model_lhs = mid_constraint.check_transformability(summaries_by_symbol,
                                                                  ivy_logic_utils.dual_clauses(proof_obligation))
            if bad_model_lhs is None:
                continue

            proof_obligation = self._generalizer.bad_model_to_proof_obligation(bad_model_lhs)
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
        transformers = filter(lambda midc: midc.rhs_pred() == predicate, self._mid_chc)
        transformability_clauses = map(lambda midc: midc.transformability_update(prestate_summaries, ivy_transrel.new),
                                       transformers)
        all_updated_syms = set.union(*(set(updated_syms) for (updated_syms, _) in transformability_clauses))
        transformability_clauses_unified = []
        for (updated_syms, clauses) in transformability_clauses:
            unchanged_equal = ivy_transrel.diff_frame(updated_syms, all_updated_syms,
                                                      im.module.relations, ivy_transrel.new)
            clauses = ivy_transrel.conjoin(clauses, unchanged_equal)
            transformability_clauses_unified.append(clauses)

        all_transformability_combined = ivy_logic_utils.or_clauses_avoid_clash(*transformability_clauses_unified)

        rhs = ivy_logic_utils.dual_clauses(lemma)
        rhs_in_new = forward_clauses(rhs, all_updated_syms)

        print "Trans: %s, check lemma: %s" % (all_transformability_combined, rhs_in_new)
        res = ivy_transrel.interpolant(all_transformability_combined,
                                       rhs_in_new,
                                       axioms=self._axioms, interpreted=None)
        assert res is not None
        return to_current_clauses(res[1], all_updated_syms)


        # # TODO: generalizing separately and then combining is potentially less efficient becauase of different local minima of the unsat core
        # lemma_generalization = ivy_logic_utils.false_clauses()
        #
        # for mid_constraint in self._mid_chc:
        #     if mid_constraint.rhs_pred() != predicate:
        #         continue
        #
        #     generalization_for_clause = mid_constraint.generalize_intransformability(prestate_summaries, lemma)
        #     # NOTE: taking care with the disjunction to rename implicitly universally quantified variables
        #     # to avoid capture between different disjuncts (each is quantified separately).
        #     # Inserting the quantifiers explicitly causes problems elsewhere, in ivy_solver.clauses_model_to_clauses
        #     lemma_generalization = ivy_logic_utils.or_clauses_avoid_clash2(lemma_generalization,
        #                                                                    generalization_for_clause)
        #
        # return lemma_generalization

#!/usr/bin/env python
from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

from pgmpy import config
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)


class HillClimbSearch(StructureEstimator):
    """
    Class for heuristic hill climb searches for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data, use_cache=True,greedy = 1, log=True, **kwargs):
        self.use_cache = use_cache
        self.tabu_list = deque(maxlen=1000)
        self.greedy = greedy # This Agent can also play the role of random exploration
        self.log = log
        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(
            self,
            model,
            score,
            structure_score,
            max_indegree,
            black_list,
            white_list,
            fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip babilistic Graphical Mp a single edge. For details on scoring
        see Koller & Friedman, Proodels, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        # tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
                set(permutations(self.variables, 2))
                - set(model.edges())
                - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                        (operation not in self.tabu_list)
                        and ((X, Y) not in black_list)
                        and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        # - score(Y, old_parents)
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        # for X, Y in model.edges():
        #     operation = ("-", (X, Y))
        #     if (operation not in self.tabu_list) and ((X, Y) not in fixed_edges):
        #         old_parents = model.get_parents(Y)
        #         new_parents = [var for var in old_parents if var != X]
        #         score_delta = score(Y, new_parents) - score(Y, old_parents)
        #         score_delta += structure_score("-")
        #         yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                    map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                        ((operation not in self.tabu_list) and ("flip", (Y, X)) not in self.tabu_list)
                        and ((X, Y) not in fixed_edges)
                        and ((Y, X) not in black_list)
                        and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                                score(X, new_X_parents)
                                + score(Y, new_Y_parents)
                                - score(X, old_X_parents)
                                - score(Y, old_Y_parents)
                        )
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

    def estimate_once(
            self,
            scoring_method="bicscore",
            start_dag=None,
            fixed_edges=set(),
            max_indegree=None,
            black_list=None,
            white_list=None,
            epsilon=1e-4,
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None

        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        Returns
        -------
        Estimated model: pgmpy.model.BayesianNetwork
            A `Bayesian Network` at a (local) score maximum.

        Examples
        --------
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        # if self.use_cache:
        if False:
            score_fn = ScoreCache.ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        # Step 1.2: Check the start_dag
        if start_dag is None:
            start_dag = BayesianNetwork()
            start_dag.add_nodes_from(self.variables)

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        # tabu_list = deque(maxlen=tabu_length)
        # tabu_list = self.tabu_list
        current_model = start_dag

        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        # operations = self._legal_operations(
        #         current_model,
        #         score_fn,
        #         score.structure_prior_ratio,
        #         max_indegree,
        #         black_list,
        #         white_list,
        #         fixed_edges,
        #     )
        # for operation in operations:
        #     print(operation)
        if self.greedy:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )
            if self.log:
                print("Non-generative state is taking a HC Step")

        else:
            action_list = []
            for i in self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
            ):
                action_list.append(i)
            if (len(action_list) > 0):
                random_index = random.randint(0, len(action_list) - 1)
                best_operation, best_score_delta = action_list[random_index]
            else:
                best_operation, best_score_delta = (None, None)

            if self.log:
                print("Non-generative state is taking a Random Step")

        # print("best delta bic: ", best_score_delta,"which is: ", best_operation, "tabu: ",self.tabu_list)

        if best_operation is None or best_score_delta < epsilon:
            # return current_model, best_operation
            return None  # None
        if best_operation[0] == "+":
            # current_model.add_edge(*best_operation[1])
            self.tabu_list.append(("-", best_operation[1]))
        # elif best_operation[0] == "-":
        #     # current_model.remove_edge(*best_operation[1])
        #     self.tabu_list.append(("+", best_operation[1]))
        elif best_operation[0] == "flip":
            X, Y = best_operation[1]
            # current_model.remove_edge(X, Y)
            # current_model.add_edge(Y, X)
            self.tabu_list.append(best_operation)

        # return current_model, best_operation
        return best_operation

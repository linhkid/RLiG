from copy import deepcopy

from pgmpy.metrics import log_likelihood_score
from tqdm import tqdm

from ..kdb import *
from ..kdb import _add_uniform
from ..utils import *
from ..structure_learning.HillClimbing import HillClimbSearch
from ..structure_learning.utils.buffer import StackBuffer
from ..structure_learning.RL_agent import ReinforcementLearningAgent
from ..structure_learning.K2_agent import K2Agent
from pgmpy.models import BayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import structure_score  # A higher score represents a better fit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx


def step(self, action):
    if action is None:
        return
    elif action[0] == "+":
        self.add_edge(*action[1])
    elif action[0] == "-":
        self.remove_edge(*action[1])
    elif action[0] == "flip":
        X, Y = action[1]
        self.remove_edge(X, Y)
        self.add_edge(Y, X)
    return


def __hash__(self):
    # 将节点和边的信息组合起来生成哈希值

    return hash((frozenset(self.nodes()), frozenset(self.edges())))


def __eq__(self, other):
    # 判断两个网络是否相等

    return isinstance(other, BayesianNetwork) and \
        frozenset(self.nodes()) == frozenset(other.nodes()) and \
        frozenset(self.edges()) == frozenset(other.edges())


BayesianNetwork.step = step
BayesianNetwork.__hash__ = __hash__
BayesianNetwork.__eq__ = __eq__


class RLiG:
    """
    The RLiG Model.
    """

    def __init__(self) -> None:
        self._d = None
        self.__gen_weights = None
        self.batch_size = None
        self.epochs = None
        self.k = None
        self.constraints = None
        self._ordinal_encoder = OrdinalEncoder(dtype=int, handle_unknown='use_encoded_value', unknown_value=-1)
        self._label_encoder = LabelEncoder()
        self.beta = 0.9
        self.beta_min = 0.1
        self.beta_decay = 0.99
        self.RLiG_Q_table = {}

    def fit(self, x, y, k=0, input_model=None, batch_size=32, episodes=2, epochs=100, warmup_epochs=1, verbose=1, gan=1,
            n=3):
        '''
        Fit the model to the given data.

        Parameters
        ----------
        x : array_like of shape (n_samples, n_features)
            Dataset to fit the model. The data should be discrete.

        y : array_like of shape (n_samples,)
            Label of the dataset.

        k : int, default=0
            Parameter k of RLiG model. Must be greater than 0. No more than 2 is Suggested.

        batch_size : int, default=32
            Size of the batch to feed the model at each step.

        epochs : int, default=0
            Number of epochs to use during training.

        warmup_epochs : int, default=1
            Number of epochs to use in warmup phase. Defaults to :attr:`1`.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.

        gan: bool, default=1
            Whether to use the discriminator to implement a gan structure. Use 1 for enabling the
            discriminator and GAN Structure

        n: int, default=4
            The definition of the n-steps in the generative states

        Returns
        -------
        self : object
            Fitted model.
        '''

        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        # self.variables = list(x.columns.values)
        self.data = pd.concat([x, y], axis=1)
        self.x = x
        self.y = y
        print(self.x)
        print(self.y)
        # self.variables = list(self.data.columns.values)
        # self.label_node = list(self.data.columns.values)[-1]
        self.variables = list(self.x.columns.values)
        self.label_node = self.y.name
        print(self.variables)
        print(self.label_node)
        x_int = self._ordinal_encoder.fit_transform(x)
        y_int = self._label_encoder.fit_transform(y).astype(int)
        # Dataframe -> Numpy.ndarry
        d = DataUtils(x_int, y_int)
        self._d = d  # DataUtils
        self.k = k  # k for kdb
        self.batch_size = batch_size

        score_buffer = StackBuffer()

        if verbose:
            print(f"warmup run:")
        history = self._warmup_run(warmup_epochs, verbose=verbose)  # 在warmup run中创建了 KDB

        # Init the Bayesian Network
        hc_agent = HillClimbSearch(data=self.data, greedy=1, log=False)  # Remove the edge deletion action
        rl_agent = K2Agent(data=self.data, label=y, greedy=0, epsilon=0.1,
                           log=False)  # Chk if there is a deletion action

        for episode in range(episodes):

            self.bayesian_network = BayesianNetwork()  # Resetting the environment
            self.bayesian_network.add_nodes_from(self.variables)
            self.bayesian_network.add_node(self.label_node)
            for variable in self.variables:
                self.bayesian_network.add_edge(self.label_node, variable)

            if input_model is not None:
                for i in range(len(self.variables)):
                    for j in range(len(self.variables)):
                        if input_model[i][j] == 1:
                            self.bayesian_network.add_edge(self.variables[i], self.variables[j])

            self.bayesian_network.fit(self.data)
            print(self.bayesian_network)
            model_graphviz = self.bayesian_network.to_graphviz()  # Testing code for graph init
            model_graphviz.draw(f"init.png", prog="dot")
            original_score = structure_score(model=self.bayesian_network, data=self.data, scoring_method="bic")
            self.best_score = original_score

            for epoch in range(epochs):
                # Parallel start
                # The distribution of n onto different CPU cores

                for i in range(n):
                    # HC Agent choose an action
                    best_action = hc_agent.estimate_once(
                        start_dag=self.bayesian_network)  # current_structure: Bayesian Network

                    next_bayesian = deepcopy(self.bayesian_network)

                    # next_bayesian take step
                    next_bayesian.step(action=best_action)
                    next_bayesian.remove_cpds(*next_bayesian.get_cpds())
                    next_bayesian.fit(self.data)
                    current_score = structure_score(model=next_bayesian, data=self.data, scoring_method="bic")

                    # Calculate the Penalty Term
                    num_parameters = len(next_bayesian.get_cpds())
                    log_n = np.log(len(self.data))
                    non_gen_complexity_penalty = (num_parameters / 2) * log_n

                    reward = (current_score - original_score) - non_gen_complexity_penalty
                    # print("reward: ", current_score, "-", original_score, "=", reward)

                    # Buffer it
                    # print("Non-gen")
                    score_buffer.push((self.bayesian_network.copy(), best_action, reward,
                                       next_bayesian.copy()))  # stackbuffer(S,A,R,S')

                    # Update the Step
                    self.bayesian_network.step(action=best_action)
                    self.bayesian_network.remove_cpds(*self.bayesian_network.get_cpds())
                    self.bayesian_network.fit(self.data)
                    original_score = current_score

                # Take a reinforcement learning step, no reward feedback now
                best_action = rl_agent.estimate_once(
                    start_dag=self.bayesian_network,
                    custom_Q_table=self.RLiG_Q_table)  # Use Reinforcement Learning agent to take a step
                current_bayesian = deepcopy(self.bayesian_network)
                self.bayesian_network.step(action=best_action)
                self.bayesian_network.remove_cpds(*self.bayesian_network.get_cpds())
                self.bayesian_network.fit(self.data)

                if (gan == 1):
                    # Feed into Ganblr and get the reward. Reward 需要归一化
                    data_sampler = BayesianModelSampling(self.bayesian_network)  # Parameters: model
                    syn_data = data_sampler.forward_sample(size=d.data_size).iloc[:, :-1]
                    print(syn_data)
                    syn_data = self._ordinal_encoder.transform(syn_data)

                    discriminator_label = np.hstack([np.ones(d.data_size), np.zeros(d.data_size)])

                # Generative State,ls is the reward, using reward to update the previous rewards
                  # Using Gan
                    discriminator_input = np.vstack([x_int, syn_data[:, :]])  # no label is included in forward_sample
                    disc_input, disc_label = sample(discriminator_input, discriminator_label, frac=0.8)
                    disc = self._discrim()
                    d_history = disc.fit(disc_input, disc_label, batch_size=batch_size, epochs=1,
                                         verbose=0).history  # discriminator fit
                    prob_fake = disc.predict(x_int, verbose=0)
                    ls = d_history['accuracy'][0]
                    print(
                        f"episode {episode}, epoch {epoch}, D_loss = {d_history['loss'][0]:.6f}, D_accuracy = {d_history['accuracy'][0]:.6f}, Q-Table Size: {len(rl_agent.Q_table)}")
                else:
                    ls = np.mean(-np.log(1))

                current_score = structure_score(model=self.bayesian_network, data=self.data,
                                                scoring_method="bic")  # NaN: Divide by zero error

                # Calculate the Penalty Term
                num_parameters = len(self.bayesian_network.get_cpds())
                log_n = np.log(len(self.data))
                gen_complexity_penalty = (num_parameters / 2) * log_n

                reward = (current_score - original_score) - gen_complexity_penalty

                original_score = current_score

                score_buffer.push((current_bayesian.copy(), best_action, reward,
                                   self.bayesian_network.copy()))  # stackbuffer(S,A,R,S')

                # Parallel end

                # Update the Q value using stack buffer
                # StackBuffer -> Update -> RL Q-Table Buffer -> Sampling -> Learn

                state, action, _, state_prime = score_buffer.stack[0]
                discount = rl_agent.gamma
                G_t = 0
                for i, (state, action, reward, state_prime) in enumerate(score_buffer.stack):
                    G_t += (reward * (1 - ls)) * (discount ** i)

                rl_agent.remember(state=deepcopy(state), action=deepcopy(action), reward=deepcopy(G_t),
                                  state_=deepcopy(state_prime))


                # Hybrid Update RL with Stack
                while not score_buffer.is_empty():
                    state, action, reward, state_prime = score_buffer.pop()
                    reward *= (1 - ls)
                    if (state, action) in rl_agent.Q_table:
                        self.RLiG_Q_table[(state, action)] = self.beta * reward + (1 - self.beta) * rl_agent.Q_table[
                            (state, action)]

            # Beta Decay
            if self.beta > self.beta_min:
                self.beta = self.beta * self.beta_decay

        return self

    def evaluate(self, x, y, model='lr') -> float:
        """
        Perform a TSTR(Training on Synthetic data, Testing on Real data) evaluation.

        Parameters
        ----------
        x, y : array_like
            Test dataset.

        model : str or object
            The model used for evaluate. Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.
            几种模型直接就是实现好的，看这里
        Return:
        --------
        accuracy_score : float.

        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import accuracy_score

        eval_model = None
        models = dict(
            lr=LogisticRegression,
            rf=RandomForestClassifier,
            mlp=MLPClassifier
        )
        if model in models.keys():
            eval_model = models[model]()
        elif hasattr(model, 'fit') and hasattr(model, 'predict'):
            eval_model = model
        else:
            raise Exception(
                "Invalid Arugument `model`, Should be one of ['lr', 'mlp', 'rf'], or a model class that have sklearn-style `fit` and `predict` method.")

        data_sampler = BayesianModelSampling(self.bayesian_network)
        syn_data = data_sampler.forward_sample(size=5000).values  # With label

        synthetic_x, synthetic_y = syn_data[:, :-1], syn_data[:, -1]
        synthetic_x = self._ordinal_encoder.transform(synthetic_x)
        synthetic_y = self._label_encoder.transform(synthetic_y)
        x_test = self._ordinal_encoder.transform(x)  # The real dataset
        y_test = self._label_encoder.transform(y)

        # Testing the model
        categories = self._d.get_categories()
        pipline = Pipeline(
            [('encoder', OneHotEncoder(categories=categories, handle_unknown='ignore')), ('model', eval_model)])
        pipline.fit(synthetic_x, synthetic_y)  # TS
        pred = pipline.predict(x_test)  # TR
        return accuracy_score(y_test, pred)

    def sample(self, size=None, verbose=1) -> np.ndarray:
        """
        Generate synthetic data.     

        Parameters
        ----------
        size : int or None
            Size of the data to be generated. set to `None` to make the size equal to the size of the training set.

        verbose : int, default=1
            Whether to output the log. Use 1 for log output and 0 for complete silence.
        
        Return:
        -----------------
        synthetic_samples : np.ndarray
            Generated synthetic data.
        """
        ordinal_data = self._sample(size, verbose)
        origin_x = self._ordinal_encoder.inverse_transform(
            ordinal_data[:, :-1])  # 因为在_sample中是按照序数编码格式，所以这一步是将序数编码格式转换为原格式
        origin_y = self._label_encoder.inverse_transform(ordinal_data[:, -1]).reshape(-1, 1)
        return np.hstack([origin_x, origin_y])

    def _sample(self, size=None, verbose=1) -> np.ndarray:
        # The sample need to be modified
        """
        Generate synthetic data in ordinal encoding format 按照序数编码格式
        """

        """
        每个节点表示一个随机变量，并且每个节点都有一个 CPD 表，用来表示该节点在其父节点给定的条件下的概率分布。具体来说，如果我们有一个节点 X
        X 和它的父节点 Parents(X)，那么 CPD 表表示 P(X∣Parents(X))。
        
        | A | B | P(C=1 | A, B) | P(C=0 | A, B) |
        |---|---|--------------|--------------|
        | 0 | 0 | 0.1 | 0.9 |
        | 0 | 1 | 0.4 | 0.6 |
        | 1 | 0 | 0.7 | 0.3 |
        | 1 | 1 | 0.9 | 0.1 |
        """
        if verbose is None or not isinstance(verbose, int):
            verbose = 1
        # basic varibles
        d = self._d  # DataUtil 应该使用数据生成的，所以要看一下DataUtil是什么
        feature_cards = np.array(d.feature_uniques)  # 特征的所有取值
        # ensure sum of each constraint group equals to 1, then re concat the probs
        _idxs = np.cumsum([0] + d._kdbe.constraints_.tolist())  # 这个应该是在生成所有的约束
        constraint_idxs = [(_idxs[i], _idxs[i + 1]) for i in range(len(_idxs) - 1)]

        # 生成CPD表
        probs = np.exp(self.__gen_weights[0])
        cpd_probs = [probs[start:end, :] for start, end in constraint_idxs]
        cpd_probs = np.vstack([p / p.sum(axis=0) for p in cpd_probs])

        # assign the probs to the full cpd tables
        idxs = np.cumsum([0] + d._kdbe.high_order_feature_uniques_)
        feature_idxs = [(idxs[i], idxs[i + 1]) for i in range(len(idxs) - 1)]
        have_value_idxs = d._kdbe.have_value_idxs_
        full_cpd_probs = []
        for have_value, (start, end) in zip(have_value_idxs, feature_idxs):
            # (n_high_order_feature_uniques, n_classes)
            cpd_prob_ = cpd_probs[start:end, :]
            # (n_all_combination) Note: the order is (*parent, variable)
            have_value_ravel = have_value.ravel()
            # (n_classes * n_all_combination)
            have_value_ravel_repeat = np.hstack([have_value_ravel] * d.num_classes)
            # (n_classes * n_all_combination) <- (n_classes * n_high_order_feature_uniques)
            full_cpd_prob_ravel = np.zeros_like(have_value_ravel_repeat, dtype=float)
            full_cpd_prob_ravel[have_value_ravel_repeat] = cpd_prob_.T.ravel()
            # (n_classes * n_parent_combinations, n_variable_unique)
            full_cpd_prob = full_cpd_prob_ravel.reshape(-1, have_value.shape[-1]).T
            full_cpd_prob = _add_uniform(full_cpd_prob, noise=0)
            full_cpd_probs.append(full_cpd_prob)

        # prepare node and edge names
        node_names = [str(i) for i in range(d.num_features + 1)]
        edge_names = [(str(i), str(j)) for i, j in d._kdbe.edges_]
        y_name = node_names[-1]

        # create TabularCPD objects
        evidences = d._kdbe.dependencies_
        feature_cpds = [
            TabularCPD(str(name), feature_cards[name], table,
                       evidence=[y_name, *[str(e) for e in evidences]],
                       evidence_card=[d.num_classes, *feature_cards[evidences].tolist()])
            for (name, evidences), table in zip(evidences.items(), full_cpd_probs)
        ]
        y_probs = (d.class_counts / d.data_size).reshape(-1, 1)
        y_cpd = TabularCPD(y_name, d.num_classes, y_probs)

        # create kDB model, then sample the data
        model = BayesianNetwork(edge_names)
        model.add_cpds(y_cpd, *feature_cpds)
        sample_size = d.data_size if size is None else size
        result = BayesianModelSampling(model).forward_sample(size=sample_size, show_progress=verbose > 0)
        sorted_result = result[node_names].values

        return sorted_result

    def _warmup_run(self, epochs, verbose=None):
        d = self._d
        tf.keras.backend.clear_session()
        ohex = d.get_kdbe_x(self.k)
        self.constraints = softmax_weight(d.constraint_positions)
        elr = get_lr(ohex.shape[1], d.num_classes, self.constraints)
        history = elr.fit(ohex, d.y, batch_size=self.batch_size, epochs=epochs, verbose=verbose)
        self.__gen_weights = elr.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _run_generator(self, loss):
        d = self._d
        ohex = d.get_kdbe_x(self.k)  # 获得一个高阶的？ Higher-order feature
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(d.num_classes, input_dim=ohex.shape[1], activation='softmax',
                                        kernel_constraint=self.constraints))
        model.compile(loss=elr_loss(loss), optimizer='adam', metrics=['accuracy'])
        model.set_weights(self.__gen_weights)
        history = model.fit(ohex, d.y, batch_size=self.batch_size, epochs=1, verbose=0)
        self.__gen_weights = model.get_weights()
        tf.keras.backend.clear_session()
        return history

    def _discrim(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(1, input_dim=self._d.num_features, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

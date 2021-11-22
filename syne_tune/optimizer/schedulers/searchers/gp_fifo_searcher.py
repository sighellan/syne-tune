# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np
from typing import Type, Optional, Dict
import copy
import logging

from syne_tune.optimizer.schedulers.searchers.searcher import \
    BaseSearcher, RandomSearcher
from syne_tune.optimizer.schedulers.searchers.gp_searcher_factory import \
    gp_fifo_searcher_factory, gp_fifo_searcher_defaults
from syne_tune.optimizer.schedulers.searchers.gp_searcher_utils import \
    DEFAULT_INITIAL_SCORING, SUPPORTED_INITIAL_SCORING, MapReward, \
    encode_state, decode_state
from syne_tune.optimizer.schedulers.searchers.utils.default_arguments \
    import check_and_merge_defaults
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.common \
    import CandidateEvaluation, Configuration, MetricValues, dictionarize_objective, INTERNAL_METRIC_NAME, \
    INTERNAL_COST_NAME
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.hp_ranges \
    import HyperparameterRanges
from syne_tune.optimizer.schedulers.searchers.bayesopt.datatypes.tuning_job_state \
    import TuningJobState
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_transformer \
    import TransformerModelFactory, ModelStateTransformer
from syne_tune.optimizer.schedulers.searchers.bayesopt.models.model_skipopt \
    import SkipOptimizationPredicate
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.base_classes \
    import LocalOptimizer, ScoringFunction, SurrogateOutputModel, \
    AcquisitionClassAndArgs, unwrap_acquisition_class_and_kwargs
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm \
    import BayesianOptimizationAlgorithm
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.bo_algorithm_components \
    import IndependentThompsonSampling
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.common \
    import RandomStatefulCandidateGenerator, ExclusionList
from syne_tune.optimizer.schedulers.searchers.bayesopt.tuning_algorithms.defaults \
    import DEFAULT_LOCAL_OPTIMIZER_CLASS, DEFAULT_NUM_INITIAL_CANDIDATES, DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS
from syne_tune.optimizer.schedulers.searchers.bayesopt.utils.duplicate_detector \
    import DuplicateDetectorIdentical
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler

logger = logging.getLogger(__name__)

__all__ = ['ModelBasedSearcher',
           'GPFIFOSearcher']


GET_CONFIG_RANDOM_RETRIES = 50


def create_initial_candidates_scorer(
        initial_scoring: str, model: SurrogateOutputModel,
        acquisition_class: AcquisitionClassAndArgs,
        random_state: np.random.RandomState,
        active_output: str = INTERNAL_METRIC_NAME) -> ScoringFunction:
    if initial_scoring == 'thompson_indep':
        if isinstance(model, Dict):
            assert active_output in model
            model = model[active_output]
        return IndependentThompsonSampling(model, random_state=random_state)
    else:
        acquisition_class, acquisition_kwargs = unwrap_acquisition_class_and_kwargs(
            acquisition_class)
        return acquisition_class(
            model, active_metric=active_output, **acquisition_kwargs)


def check_initial_candidates_scorer(initial_scoring: str) -> str:
    if initial_scoring is None:
        return DEFAULT_INITIAL_SCORING
    else:
        assert initial_scoring in SUPPORTED_INITIAL_SCORING, \
            "initial_scoring = '{}' is not supported".format(
                initial_scoring)
        return initial_scoring


class ModelBasedSearcher(BaseSearcher):
    """Common code for surrogate model based searchers

    """
    def __init__(
            self, configspace, metric=None,
            points_to_evaluate=None):
        super().__init__(configspace, metric, points_to_evaluate)

    def _create_internal(
            self, hp_ranges: HyperparameterRanges, random_seed: int,
            model_factory: TransformerModelFactory,
            acquisition_class: AcquisitionClassAndArgs,
            map_reward: Optional[MapReward] = None,
            init_state: TuningJobState = None,
            local_minimizer_class: Type[LocalOptimizer] = None,
            skip_optimization: SkipOptimizationPredicate = None,
            num_initial_candidates: int = DEFAULT_NUM_INITIAL_CANDIDATES,
            num_initial_random_choices: int = DEFAULT_NUM_INITIAL_RANDOM_EVALUATIONS,
            initial_scoring: Optional[str] = None,
            cost_attr: Optional[str] = None,
            resource_attr: Optional[str] = None):
        self.hp_ranges = hp_ranges
        self.random_seed = random_seed
        self.num_initial_candidates = num_initial_candidates
        self.num_initial_random_choices = num_initial_random_choices
        self.map_reward = map_reward
        if local_minimizer_class is None:
            self.local_minimizer_class = DEFAULT_LOCAL_OPTIMIZER_CLASS
        else:
            self.local_minimizer_class = local_minimizer_class
        self.acquisition_class = acquisition_class
        self._debug_log = model_factory.debug_log
        self.initial_scoring = check_initial_candidates_scorer(initial_scoring)
        # Create state transformer
        # Initial state is empty (note that the state is mutable)
        if init_state is None:
            init_state = TuningJobState(
                hp_ranges=self._hp_ranges_in_state(),
                candidate_evaluations=[],
                failed_candidates=[],
                pending_evaluations=[])
        self.state_transformer = ModelStateTransformer(
            model_factory=model_factory,
            init_state=init_state,
            skip_optimization=skip_optimization)
        self.random_state = np.random.RandomState(random_seed)
        self.random_generator = RandomStatefulCandidateGenerator(
            self._hp_ranges_for_prediction(), random_state=self.random_state)
        self.set_profiler(model_factory.profiler)
        self._cost_attr = cost_attr
        self._resource_attr = resource_attr
        self._random_searcher = None
        if model_factory.debug_log is not None:
            deb_msg = "[ModelBasedSearcher.__init__]\n"
            deb_msg += ("- acquisition_class = {}\n".format(acquisition_class))
            deb_msg += ("- local_minimizer_class = {}\n".format(local_minimizer_class))
            deb_msg += ("- num_initial_candidates = {}\n".format(num_initial_candidates))
            deb_msg += ("- num_initial_random_choices = {}\n".format(num_initial_random_choices))
            deb_msg += ("- initial_scoring = {}\n".format(self.initial_scoring))
            logger.info(deb_msg)

    def _copy_kwargs_to_kwargs_int(self, kwargs_int: Dict, kwargs: Dict):
        # Extra arguments not parsed in factory
        for k in ('init_state', 'local_minimizer_class', 'cost_attr',
                  'resource_attr'):
            kwargs_int[k] = kwargs.get(k)

    def _hp_ranges_in_state(self):
        """
        :return: HyperparameterRanges to be used in self.state_transformer.state
        """
        return self.hp_ranges

    def _hp_ranges_for_prediction(self):
        """
        :return: HyperparameterRanges to be used in predictions and acquisition
            functions
        """
        return self._hp_ranges_in_state()

    def _metric_val_update(
            self, config: Dict, crit_val: float, result: Dict) -> MetricValues:
        return crit_val

    def on_trial_result(
            self, trial_id: str, config: Dict, result: Dict, update: bool):
        # If both `cost_attr` and `resource_attr` are given, cost data (if
        # given) is written out from every `result`, independent of `update`
        cattr = self._cost_attr
        rattr = self._resource_attr
        if cattr is not None and rattr is not None and cattr in result \
                and rattr in result:
            cost_val = float(result[cattr])
            resource = str(result[rattr])
            metrics = {INTERNAL_COST_NAME: {resource: cost_val}}
            self.state_transformer.label_candidate(CandidateEvaluation(
                candidate=config, metrics=metrics))
        if update:
            self._update(trial_id, config, result)

    def _trial_id_string(self, trial_id: str, result: Dict):
        """
        For multi-fidelity, we also want to output the resource level
        """
        return trial_id

    def _update(self, trial_id: str, config: Dict, result: Dict):
        metric_val = result[self._metric]
        if self.map_reward is not None:
            crit_val = self.map_reward(metric_val)
        else:
            crit_val = metric_val
        metrics = dictionarize_objective(
            self._metric_val_update(config, crit_val, result))
        # Cost value only dealt with here if `resource_attr` not given
        attr = self._cost_attr
        cost_val = None
        if attr is not None and attr in result:
            cost_val = float(result[attr])
            if self._resource_attr is None:
                metrics[INTERNAL_COST_NAME] = cost_val
        self.state_transformer.label_candidate(CandidateEvaluation(
            candidate=config, metrics=metrics))
        if self.debug_log is not None:
            _trial_id = self._trial_id_string(trial_id, result)
            msg = f"Update for trial_id {_trial_id}: metric = {metric_val:.3f}"
            if self.map_reward is not None:
                msg += f", crit_val = {crit_val:.3f}"
            if cost_val is not None:
                msg += f", cost_val = {cost_val:.2f}"
            logger.info(msg)

    def _get_config_modelbased(self, exclusion_candidates, **kwargs) -> \
            Optional[Configuration]:
        raise NotImplementedError()

    def _get_exclusion_candidates(self, **kwargs) -> ExclusionList:
        return ExclusionList(self.state_transformer.state)

    def get_config(self, **kwargs) -> Configuration:
        """
        Runs Bayesian optimization in order to suggest the next config to evaluate.

        :return: Next config to evaluate at
        """
        self._assign_random_searcher()
        trial_id = kwargs.get('trial_id')
        state = self.state_transformer.state
        if self.do_profile:
            # Start new profiler block
            meta = {
                'fit_hyperparams': not self.state_transformer.skip_optimization(
                    state),
                'num_observed': len(state.candidate_evaluations),
                'num_pending': len(state.pending_evaluations)
            }
            self.profiler.begin_block(meta)
            self.profiler.start('all')
        config = self._next_initial_config()  # Ask for initial config
        exclusion_candidates = None
        if config is None:
            exclusion_candidates = self._get_exclusion_candidates(**kwargs)
            pick_random = (len(exclusion_candidates) < self.num_initial_random_choices) or \
                (not state.candidate_evaluations)
        else:
            pick_random = True  # Initial configs count as random here
        if self.debug_log is not None:
            self.debug_log.start_get_config(
                'random' if pick_random else 'BO', trial_id=trial_id)
        if config is None:
            if pick_random:
                if self.do_profile:
                    self.profiler.start('random')
                for _ in range(GET_CONFIG_RANDOM_RETRIES):
                    _config = self._random_searcher.get_config()
                    if _config is None:
                        # If `RandomSearcher` returns no config at all, the
                        # search space is exhausted
                        break
                    if not exclusion_candidates.contains(_config):
                        config = _config
                        break
                if self.do_profile:
                    self.profiler.stop('random')
            elif not exclusion_candidates.config_space_exhausted():
                config = self._get_config_modelbased(
                    exclusion_candidates, **kwargs)

        if config is not None:
            if self.debug_log is not None:
                self.debug_log.set_final_config(config, trial_id=trial_id)
                # All get_config debug log info is only written here
                self.debug_log.write_block()
        else:
            msg = "Failed to sample a configuration not already chosen " + \
                  f"before. Exclusion list has size {len(exclusion_candidates)}."
            cs_size = exclusion_candidates.configspace_size
            if cs_size is not None:
                msg += f" Configuration space has size {cs_size}."
            logger.warning(msg)
        if self.do_profile:
            self.profiler.stop('all')
            self.profiler.clear()

        return config

    def dataset_size(self):
        return len(self.state_transformer.state.candidate_evaluations)

    def model_parameters(self):
        return self.state_transformer.get_params()

    def set_params(self, param_dict):
        self.state_transformer.set_params(param_dict)

    def get_state(self):
        """
        The mutable state consists of the GP model parameters, the
        TuningJobState, and the skip_optimization predicate (which can have a
        mutable state).
        We assume that skip_optimization can be pickled.

        """
        state = {
            'model_params': self.model_parameters(),
            'state': encode_state(self.state_transformer.state),
            'skip_optimization': self.state_transformer.skip_optimization,
            'random_state': self.random_state}
        if self.debug_log is not None:
            state['debug_log'] = self.debug_log.get_mutable_state()
        if self._random_searcher is not None:
            state['random_searcher_state'] = self._random_searcher.get_state()
        return state

    def _clone_from_state_common(
            self, new_searcher: 'ModelBasedSearcher', state: Dict):
        new_searcher.state_transformer.set_params(state['model_params'])
        new_searcher.random_state = state['random_state']
        new_searcher.random_generator.random_state = \
            state['random_state']
        if self.debug_log and 'debug_log' in state:
            new_searcher.debug_log.set_mutable_state(state['debug_log'])
        if 'random_searcher_state' in state:
            # Restore self._random_searcher as well
            self._assign_random_searcher()
            self._random_searcher._restore_from_state(
                state['random_searcher_state'])

    def set_profiler(self, profiler: Optional[SimpleProfiler]):
        self.profiler = profiler
        self.do_profile = (profiler is not None)

    @property
    def debug_log(self):
        return self._debug_log

    def _assign_random_searcher(self):
        if self._random_searcher is None:
            # Used for initial random configs (if any)
            # We do not have to deal with points_to_evaluate
            self._random_searcher = RandomSearcher(
                self.hp_ranges.config_space,
                metric=self._metric,
                points_to_evaluate=[],
                random_seed=self.random_seed,
                debug_log=False)


class GPFIFOSearcher(ModelBasedSearcher):
    """Gaussian process Bayesian optimization for FIFO scheduler

    This searcher must be used with `FIFOScheduler`. It provides Bayesian
    optimization, based on a Gaussian process surrogate model.

    Pending configurations (for which evaluation tasks are currently running)
    are dealt with by fantasizing (i.e., target values are drawn from the
    current posterior, and acquisition functions are averaged over this
    sample, see `num_fantasy_samples`).
    The GP surrogate model uses a Matern 5/2 covariance function with automatic
    relevance determination (ARD) of input attributes, and a constant mean
    function. The acquisition function is expected improvement (EI). All
    hyperparameters of the surrogate model are estimated by empirical Bayes
    (maximizing the marginal likelihood). In general, this hyperparameter
    fitting is the most expensive part of a `get_config` call.

    The following happens in `get_config`. For the first `num_init_random` calls,
    a config is drawn at random (the very first call results in the default
    config of the space). Afterwards, Bayesian optimization is used, unless
    there are no finished evaluations yet.
    First, model hyperparameter are refit. This step can be skipped (see
    `opt_skip*` parameters). Next, `num_init_candidates` configs are sampled at
    random, and ranked by a scoring function (`initial_scoring`). BFGS local
    optimization is then run starting from the top scoring config, where EI
    is minimized.

    Parameters
    ----------
    configspace : Dict
        Configuration space. Constant parameters are filtered out
    metric : str
        Name of metric reported by evaluation function.
    points_to_evaluate: List[Dict] or None
        List of configurations to be evaluated initially (in that order).
        Each config in the list can be partially specified, or even be an
        empty dict. For each hyperparameter not specified, the default value
        is determined using a midpoint heuristic.
        If None (default), this is mapped to [dict()], a single default config
        determined by the midpoint heuristic. If [] (empty list), no initial
        configurations are specified.
    debug_log : bool (default: False)
        If True, both searcher and scheduler output an informative log, from
        which the configs chosen and decisions being made can be traced.
    resource_attr : str (optional)
        Name of resource attribute in reports. This is optional here, but
        required for multi-fidelity searchers.
        If `resource_attr` and `cost_attr` are given, cost values are read from
        each report and stored in the state. This allows cost models to be fit
        on more data.
    cost_attr : str (optional)
        Name of cost attribute in data obtained from reporter (e.g., elapsed
        training time). Needed only by cost-aware searchers. Depending on
        whether `resource_attr` is given, cost values are read from each
        report or only at the end.
    random_seed : int
        Seed for pseudo-random number generator used.
    num_init_random : int
        Number of initial `get_config` calls for which randomly sampled configs
        are returned. Afterwards, Bayesian optimization is used
    num_init_candidates : int
        Number of initial candidates sampled at random in order to seed the
        search for `get_config`
    num_fantasy_samples : int
        Number of samples drawn for fantasizing (latent target values for
        pending candidates)
    initial_scoring : str
        Scoring function to rank initial candidates (local optimization of EI
        is started from top scorer). Values are 'thompson_indep' (independent
        Thompson sampling; randomized score, which can increase exploration),
        'acq_func' (score is the same (EI) acquisition function which is afterwards
        locally optimized).
    opt_nstarts : int
        Parameter for hyperparameter fitting. Number of random restarts
    opt_maxiter : int
        Parameter for hyperparameter fitting. Maximum number of iterations
        per restart
    opt_warmstart : bool
        Parameter for hyperparameter fitting. If True, each fitting is started
        from the previous optimum. Not recommended in general
    opt_verbose : bool
        Parameter for hyperparameter fitting. If True, lots of output
    opt_skip_init_length : int
        Parameter for hyperparameter fitting, skip predicate. Fitting is never
        skipped as long as number of observations below this threshold
    opt_skip_period : int
        Parameter for hyperparameter fitting, skip predicate. If >1, and number
        of observations above `opt_skip_init_length`, fitting is done only
        K-th call, and skipped otherwise
    map_reward : str or MapReward
        If `mode == 'max'`, the scheduler maximizes reward, while
        internally, Bayesian optimization is minimizing the criterion. States
        how reward is mapped to criterion. If the mode is 'min', this
        argument is ignored.
        Built-in are `minus_x` (criterion = -reward) and `<a>_minus_x`, where
        <a> is a constant (criterion = <a> - reward), for example `1_minus_x`.
        From a technical standpoint, it does not matter what is chosen here,
        because criterion is only used internally. Also note that criterion
        data is always normalized to mean 0, variance 1 before fitted with a
        GP.
    """
    def __init__(self, configspace, **kwargs):
        if configspace is not None:
            super().__init__(
                configspace, metric=kwargs.get('metric'),
                points_to_evaluate=kwargs.get('points_to_evaluate'))
            kwargs['configspace'] = configspace
            kwargs_int = self._create_kwargs_int(kwargs)
        else:
            # Internal constructor, bypassing the factory
            kwargs_int = kwargs.copy()
        self._call_create_internal(**kwargs_int)

    def _create_kwargs_int(self, kwargs):
        _kwargs = check_and_merge_defaults(
            kwargs, *gp_fifo_searcher_defaults(),
            dict_name='search_options')
        # Extra arguments not parsed in factory
        kwargs_int = gp_fifo_searcher_factory(**_kwargs)
        self._copy_kwargs_to_kwargs_int(kwargs_int, kwargs)
        return kwargs_int

    def _call_create_internal(self, **kwargs_int):
        """
        Part of constructor which can be different in subclasses
        """
        self._create_internal(**kwargs_int)

    def configure_scheduler(self, scheduler):
        from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

        assert isinstance(scheduler, FIFOScheduler), \
            "This searcher requires FIFOScheduler scheduler"
        super().configure_scheduler(scheduler)

    def register_pending(self, config: Configuration, milestone=None):
        """
        Registers config as pending. This means the corresponding evaluation
        task is running. Once it finishes, update is called for config.

        """
        # It is OK for the candidate already to be registered as pending, in
        # which case we do nothing
        state = self.state_transformer.state
        if config not in state.pending_candidates:
            pos_cand = state.pos_of_config(config)
            if pos_cand is not None:
                evals = state.candidate_evaluations
                num_labeled = len(evals)
                error_msg = """
                This configuration is already registered as labeled:
                   Position of labeled candidate: {} of {}
                   Label value: {}
                """.format(
                    pos_cand, num_labeled,
                    evals[pos_cand].metrics[INTERNAL_METRIC_NAME])
                assert False, error_msg
            self.state_transformer.append_candidate(config)

    def _get_config_modelbased(self, exclusion_candidates, **kwargs) -> \
            Optional[Configuration]:
        state = self.state_transformer.state
        # Obtain current SurrogateModel from state transformer. Based on
        # this, the BO algorithm components can be constructed
        if self.do_profile:
            self.profiler.push_prefix('getconfig')
            self.profiler.start('all')
            self.profiler.start('gpmodel')
        # Note: Asking for the model triggers the posterior computation
        model = self.state_transformer.model()
        if self.do_profile:
            self.profiler.stop('gpmodel')
        # Create BO algorithm
        initial_candidates_scorer = create_initial_candidates_scorer(
            self.initial_scoring, model, self.acquisition_class,
            self.random_state)
        local_optimizer = self.local_minimizer_class(
            hp_ranges=state.hp_ranges,
            model=model,
            acquisition_class=self.acquisition_class,
            active_metric=INTERNAL_METRIC_NAME)
        bo_algorithm = BayesianOptimizationAlgorithm(
            initial_candidates_generator=self.random_generator,
            initial_candidates_scorer=initial_candidates_scorer,
            num_initial_candidates=self.num_initial_candidates,
            local_optimizer=local_optimizer,
            pending_candidate_state_transformer=None,
            exclusion_candidates=exclusion_candidates,
            num_requested_candidates=1,
            greedy_batch_selection=False,
            duplicate_detector=DuplicateDetectorIdentical(),
            profiler=self.profiler,
            sample_unique_candidates=False,
            debug_log=self.debug_log)
        # Next candidate decision
        _config = bo_algorithm.next_candidates()
        if len(_config) > 0:
            config = _config[0]
        else:
            config = None
        if self.do_profile:
            self.profiler.stop('all')
            self.profiler.pop_prefix()  # getconfig
        return config

    def evaluation_failed(self, config: Configuration):
        # Remove pending candidate
        self.state_transformer.drop_pending_candidate(config)
        # Mark config as failed (which means it will be blacklisted in
        # future get_config calls)
        self.state_transformer.mark_candidate_failed(config)

    def clone_from_state(self, state):
        # Create clone with mutable state taken from 'state'
        init_state = decode_state(state['state'], self.hp_ranges)
        skip_optimization = state['skip_optimization']
        model_factory = self.state_transformer._model_factory
        if isinstance(model_factory, dict):
            # this is a single-key dictionary, so we extract the only value it contains
            model_factory = next(iter(model_factory.values()))
        if isinstance(skip_optimization, dict):
            # this is a single-key dictionary, so we extract the only value it contains
            skip_optimization = next(iter(skip_optimization.values()))
        # Call internal constructor
        new_searcher = GPFIFOSearcher(
            configspace=None,
            hp_ranges=self.hp_ranges,
            random_seed=self.random_seed,
            model_factory=model_factory,
            acquisition_class=self.acquisition_class,
            map_reward=self.map_reward,
            init_state=init_state,
            local_minimizer_class=self.local_minimizer_class,
            skip_optimization=skip_optimization,
            num_initial_candidates=self.num_initial_candidates,
            num_initial_random_choices=self.num_initial_random_choices,
            initial_scoring=self.initial_scoring,
            cost_attr=self._cost_attr,
            resource_attr=self._resource_attr)
        self._clone_from_state_common(new_searcher, state)
        # Invalidate self (must not be used afterwards)
        self.state_transformer = None
        return new_searcher
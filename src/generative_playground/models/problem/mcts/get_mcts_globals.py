from generative_playground.models.problem.mcts.result_repo import RuleChoiceRepository, ExperienceRepository, to_bins
from .model_adapters import MCTSRewardProcessor
from generative_playground.molecules.models.conditional_probability_model import CondtionalProbabilityModel
from generative_playground.models.problem.mcts.model_adapters import optimizer_factory_gen
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
from generative_playground.codec.codec import get_codec
from generative_playground.utils.gpu_utils import device


# TODO: merge with get_globals


def get_thompson_globals(num_bins=50,  # TODO: replace with a Value Distribution object
                         reward_fun_=None,
                         grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
                         max_seq_length=60,
                         decay=0.95,
                         updates_to_refresh=10):
    grammar_name = 'hypergraph:' + grammar_cache
    codec = get_codec(True, grammar_name, max_seq_length)
    reward_proc = RewardProcessor(num_bins)

    rule_choice_repo_factory = lambda x: RuleChoiceRepository(reward_proc=reward_proc,
                                                              mask=x,
                                                              decay=decay)

    exp_repo_ = ExperienceRepository(grammar=codec.grammar,
                                     reward_preprocessor=reward_proc,
                                     decay=decay,
                                     conditional_keys=[key for key in codec.grammar.conditional_frequencies.keys()],
                                     rule_choice_repo_factory=rule_choice_repo_factory)

    # TODO: weave this into the nodes to do node-level action averages as regularization
    local_exp_repo_factory = lambda graph: ExperienceRepository(grammar=codec.grammar,
                                                                reward_preprocessor=reward_proc,
                                                                decay=decay,
                                                                conditional_keys=[i for i in range(len(graph))],
                                                                rule_choice_repo_factory=rule_choice_repo_factory)

    globals = GlobalParametersThompson(codec.grammar,
                                       max_seq_length,
                                       exp_repo_,
                                       decay=decay,
                                       updates_to_refresh=updates_to_refresh,
                                       reward_fun=reward_fun_,
                                       reward_proc=reward_proc,
                                       rule_choice_repo_factory=rule_choice_repo_factory,
                                       state_store=None
                                       )

    return globals


class RewardProcessor:
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def __call__(self, x):
        return (1, to_bins(x, self.num_bins))


class GlobalParametersParent:
    def __init__(self,
                 grammar,
                 max_depth,
                 reward_fun=None,
                 state_store={},
                 plotter=None):
        self.grammar = grammar
        self.max_depth = max_depth
        self.reward_fun = reward_fun
        self.state_store = state_store
        self.plotter = plotter


class GlobalParametersModel(GlobalParametersParent):
    def __init__(self,
                 batch_size=20,
                 reward_fun_=None,
                 grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
                 max_depth=60,
                 lr=0.05,
                 grad_clip=5,
                 entropy_weight=3,
                 decay=None,
                 num_bins=None,
                 updates_to_refresh=None,
                 plotter=None
                 ):
        grammar_name = 'hypergraph:' + grammar_cache
        codec = get_codec(True, grammar_name, max_depth)
        super().__init__(codec.grammar, max_depth, reward_fun_, {}, plotter=plotter)
        do_model = True
        if do_model:
            # create optimizer factory
            optimizer_factory = optimizer_factory_gen(lr, grad_clip)
            # create model
            model = CondtionalProbabilityModel(codec.grammar).to(device)
            # create loss object
            loss_type = 'advantage_record'
            loss_fun = PolicyGradientLoss(loss_type, entropy_wgt=entropy_weight)
            self.model = model
            self.process_reward = MCTSRewardProcessor(loss_fun, model, optimizer_factory, batch_size)
        else:
            self.model = lambda x, y: y

        self.decay = decay
        self.reward_proc = RewardProcessor(num_bins)
        self.updates_to_refresh = updates_to_refresh

    def get_mutable_state(self):
        state = {'model': self.model,
                 'plotter': self.plotter,
                 }
        if hasattr(self.reward_fun, 'history_data'):
            state['history_data'] = self.reward_fun.history_data
        return state

    def set_mutable_state(self, state):
        self.model = state['model']
        self.process_reward.model = state['model']
        self.plotter = state['plotter']
        if 'history_data' in state:
            self.reward_fun.history_data = state['history_data']


class GlobalParametersThompson(GlobalParametersParent):
    def __init__(self,
                 grammar,
                 max_depth,
                 exp_repo,
                 decay=0.99,
                 updates_to_refresh=100,
                 reward_fun=None,
                 reward_proc=None,
                 rule_choice_repo_factory=None,
                 num_bins=50,
                 state_store={},
                 plotter=None):
        super().__init__(grammar, max_depth, reward_fun, state_store, plotter=plotter)
        self.experience_repository = exp_repo
        self.decay = decay
        self.updates_to_refresh = updates_to_refresh
        self.reward_proc = reward_proc
        self.rule_choice_repo_factory = rule_choice_repo_factory
        self.num_bins = num_bins

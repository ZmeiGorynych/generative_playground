from .model_adapters import MCTSRewardProcessor
from generative_playground.molecules.models.conditional_probability_model import CondtionalProbabilityModel
from generative_playground.models.problem.mcts.model_adapters import optimizer_factory_gen
from generative_playground.models.losses.policy_gradient_loss import PolicyGradientLoss
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
from generative_playground.codec.codec import get_codec
from generative_playground.models.problem.mcts.node import GlobalParametersModel
from generative_playground.utils.gpu_utils import device

# TODO: merge with get_globals
def get_model_globals(batch_size=20,
                      reward_fun_=None,
                      grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
                      max_depth=60,
                      lr=0.05,
                      grad_clip=5,
                      entropy_weight=1):
    grammar_name = 'hypergraph:' + grammar_cache
    codec = get_codec(True, grammar_name, max_depth)
    # create optimizer factory
    optimizer_factory = optimizer_factory_gen(lr, grad_clip)
    # create model
    model = CondtionalProbabilityModel(codec.grammar).to(device)
    # create loss object
    loss_type = 'advantage_record'
    loss_fun = PolicyGradientLoss(loss_type, entropy_wgt=entropy_weight)
    process_reward = MCTSRewardProcessor(loss_fun, model, optimizer_factory, batch_size)
    globals =  GlobalParametersModel(
                 codec.grammar,
                 max_depth,
                 reward_fun=reward_fun_,
                 state_store={},
                 model=model,
                 process_reward=process_reward
                 )
    return globals



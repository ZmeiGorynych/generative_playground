# have to monkey-patch shelve to replace pickle with dill, for support of saving lambdas
import tempfile
from generative_playground.data_utils.shelve import Shelve
from generative_playground.utils.persistent_dict import PersistentDict
from generative_playground.molecules.visualize_molecules import model_process_fun
from generative_playground.codec.hypergraph_mask_generator import *
from generative_playground.codec.codec import get_codec
from generative_playground.models.problem.mcts.node import GlobalParameters, \
    MCTSNodeLocalThompson, MCTSNodeGlobalThompson
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.models.problem.mcts.result_repo import ExperienceRepository, to_bins, \
    RuleChoiceRepository
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions


def explore(root_node, num_sims):
    rewards = []
    infos = []
    for _ in range(num_sims):
        next_node = root_node
        while True:
            probs = next_node.action_probabilities()
            action = np.random.multinomial(1, probs).argmax()
            next_node, reward, info = next_node.apply_action(action)
            is_terminal = next_node.is_terminal()
            if is_terminal:
                next_node.back_up(reward)
                rewards.append(reward)
                infos.append(info)
                break
    return rewards, infos

class RewardProcessor:
    def __init__(self, num_bins):
        self.num_bins = num_bins

    def __call__(self, x):
        return (1, to_bins(x, self.num_bins))

def get_globals(num_bins=50,  # TODO: replace with a Value Distribution object
                  ver='trivial',  # 'v2'#'
                  obj_num=4,
                  grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
                  max_seq_length=60,
                  decay=0.95):
    reward_fun_ = guacamol_goal_scoring_functions(ver)[obj_num]
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

    globals = GlobalParameters(codec.grammar,
                               max_seq_length,
                               exp_repo_,
                               decay=decay,
                               updates_to_refresh=100,
                               reward_fun=reward_fun_,
                               reward_proc=reward_proc,
                               rule_choice_repo_factory=rule_choice_repo_factory,
                               state_store=None
                               )




    return globals


def run_mcts(num_batches=10000,
             num_bins=50,  # TODO: replace with a Value Distribution object
             ver='trivial',  # 'v2'#'
             obj_num=4,
             grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
             max_seq_length=60,
             decay=0.95,
             node_type=MCTSNodeGlobalThompson,
             dashboard_name='',
             compress_data_store=True
             ):
    plotter = MetricPlotter(plot_prefix='',
                            save_file=None,
                            loss_display_cap=4,
                            dashboard_name=dashboard_name,
                            plot_ignore_initial=0,
                            process_model_fun=model_process_fun,
                            extra_metric_fun=None,
                            smooth_weight=0.5,
                            frequent_calls=False)

    my_globals = get_globals(num_bins=num_bins,
                                       ver=ver,
                                       obj_num=obj_num,
                                       grammar_cache=grammar_cache,
                                       max_seq_length=max_seq_length,
                                       decay=decay)

    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    db_path = '/{}/kv_store.db'.format(temp_dir_path)
    with Shelve(db_path, 'kv_table', compress=compress_data_store) as state_store:
        my_globals.state_store = state_store
        root_node = node_type(my_globals,
                              parent=None,
                              source_action=None,
                              depth=1)

        for _ in range(num_batches):
            rewards, infos = explore(root_node, 20)
            state_store.flush()
            # visualisation code goes here
            plotter_input = {'rewards': np.array(rewards),
                             'info': [[x['smiles'] for x in infos], np.ones(len(infos))]}
            plotter(None, None, plotter_input, None, None)
            print(max(rewards))

    print(root_node.result_repo.avg_reward())
    temp_dir.cleanup()
    print("done!")

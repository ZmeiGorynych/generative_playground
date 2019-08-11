# have to monkey-patch shelve to replace pickle with dill, for support of saving lambdas
import tempfile
from generative_playground.data_utils.shelve import Shelve
from generative_playground.utils.persistent_dict import PersistentDict
from generative_playground.codec.hypergraph_mask_generator import *
from generative_playground.codec.codec import get_codec
from generative_playground.models.problem.mcts.node import GlobalParameters, \
    MCTSNodeLocalThompson, MCTSNodeGlobalThompson
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


# def child_rewards_to_log_probs(child_rewards):
#     total = 0
#     counter = 0
#     for c in child_rewards:
#         if c is not None:
#             total += c
#             counter += 1
#     if counter == 0:
#         return np.zeros((len(child_rewards)))
#     avg_reward = total / counter
#     for c in range(len(child_rewards)):
#         if child_rewards[c] is None:
#             child_rewards[c] = avg_reward
#
#     log_probs, probs = log_thompson_probabilities(np.array(child_rewards))
#     return log_probs


# linked tree with nodes


if __name__ == '__main__':
    shelve_fn = 'states'
    num_bins = 50  # TODO: replace with a Value Distribution object
    ver = 'trivial'
    obj_num = 4
    reward_fun_ = guacamol_goal_scoring_functions(ver)[obj_num]
    grammar_cache = 'hyper_grammar_guac_10k_with_clique_collapse.pickle'  # 'hyper_grammar.pickle'
    grammar_name = 'hypergraph:' + grammar_cache
    max_seq_length = 30
    num_batches = 100
    decay = 0.99
    codec = get_codec(True, grammar_name, max_seq_length)
    reward_proc = lambda x: (1, to_bins(x, num_bins))
    exp_repo_ = ExperienceRepository(grammar=codec.grammar,
                                     reward_preprocessor=reward_proc,
                                     decay=decay)



    rule_choice_repo_factory = lambda x: RuleChoiceRepository(reward_proc=reward_proc,
                                                    mask=x,
                                                    decay=decay)

    state_store = {}
    globals = GlobalParameters(codec.grammar,
                               max_seq_length,
                               exp_repo_,
                               decay=decay,
                               updates_to_refresh=100,
                               reward_fun=reward_fun_,
                               reward_proc=reward_proc,
                               rule_choice_repo_factory=rule_choice_repo_factory,
                               state_store=state_store
                               )


    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = temp_dir.name
    db_path = '/{}/kv_store.db'.format(temp_dir_path)
    with Shelve(db_path, 'kv_table') as state_store:
        globals.state_store = state_store
        root_node = MCTSNodeLocalThompson(globals,
                             parent=None,
                             source_action=None,
                             depth=1)
        state_store.flush()

        for _ in range(num_batches):
            rewards, infos = explore(root_node, 100)
            state_store.flush()
            # visualisation code goes here
            print(max(rewards))

    print(root_node.result_repo.avg_reward())
    temp_dir.cleanup()
    print("done!")

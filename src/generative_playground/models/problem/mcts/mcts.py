# have to monkey-patch shelve to replace pickle with dill, for support of saving lambdas
import os
import gzip, dill
from generative_playground.data_utils.shelve import Shelve
from generative_playground.molecules.visualize_molecules import model_process_fun
from generative_playground.codec.hypergraph_mask_generator import *
from generative_playground.models.problem.mcts.node import MCTSNodeLocalThompson, MCTSNodeGlobalThompson, \
    MCTSNodeGlobalModel, MCTSNodeLocalModel, MCTSNodeGlobalModelLocalThompson
from generative_playground.metrics.metric_monitor import MetricPlotter
from generative_playground.molecules.guacamol_utils import guacamol_goal_scoring_functions
from generative_playground.models.problem.mcts.get_mcts_globals import get_thompson_globals, GlobalParametersModel
from generative_playground.models.reward_adjuster import CountRewardAdjuster


def explore(root_node, num_sims):
    rewards = []
    infos = []
    for _ in range(num_sims):
        next_node = root_node
        while True:
            probs = next_node.action_probabilities().astype('float64')
            probs /= sum(probs)
            pre_action = np.random.multinomial(1, probs)
            action = pre_action.argmax()
            next_node, reward, info = next_node.apply_action(action)
            is_terminal = next_node.is_terminal()
            if is_terminal:
                next_node.back_up(reward)
                rewards.append(reward)
                infos.append(info)
                break
    return rewards, infos


def class_from_kind(kind):
    if kind == 'thompson_local':
        return MCTSNodeLocalThompson
    elif kind == 'thompson_global':
        return MCTSNodeGlobalThompson
    if kind == 'model_global':
        return MCTSNodeGlobalModel
    if kind == 'model_local':
        return MCTSNodeLocalModel
    if kind == 'model_mixed':
        return MCTSNodeGlobalModelLocalThompson


def run_mcts(num_batches=50, # respawn after that - workaround for memory leak
             batch_size=20,
             ver='trivial',  # 'v2'#'
             obj_num=4,
             grammar_cache='hyper_grammar_guac_10k_with_clique_collapse.pickle',  # 'hyper_grammar.pickle'
             max_seq_length=60,
             base_name='',
             compress_data_store=True,
             kind='thompson_local',
             reset_cache=False,

             num_bins=50,  # TODO: replace with a Value Distribution object
             updates_to_refresh=10,
             decay=0.95,

             lr=0.05,
             grad_clip=5,
             entropy_weight=3,
             ):
    root_name = base_name + '_' + ver + '_' + str(obj_num)

    pre_reward_fun = lambda x: guacamol_goal_scoring_functions(ver)[obj_num]([x])[0]
    if False:  # 'model' in kind:
        reward_fun_ = CountRewardAdjuster(pre_reward_fun)
    else:
        reward_fun_ = pre_reward_fun

    plotter = MetricPlotter(plot_prefix='',
                            save_file=None,
                            loss_display_cap=4,
                            dashboard_name=None,  # root_name,
                            plot_ignore_initial=0,
                            process_model_fun=model_process_fun,
                            extra_metric_fun=None,
                            smooth_weight=0.5,
                            frequent_calls=False)

    # load or create the global variables needed
    here = os.path.dirname(__file__)
    save_path = os.path.realpath(here + '../../../../molecules/train/mcts/data/') + '/'
    globals_name = os.path.realpath(save_path + root_name + '.gpkl')
    db_path = '/' + os.path.realpath(save_path + root_name + '.db').replace('\\', '/')
    if reset_cache:
        try:
            os.remove(globals_name)
            print('removed globals cache ' + globals_name)
        except:
            print("Could not remove globals cache" + globals_name)

        try:
            os.remove(db_path[1:])
            print('removed locals cache ' + db_path[1:])
        except:
            print("Could not remove locals cache" + db_path[1:])

    try:
        with gzip.open(globals_name) as f:
            my_globals = dill.load(f)
    except:
        if 'thompson' in kind:
            my_globals = get_thompson_globals(num_bins=num_bins,
                                              reward_fun_=reward_fun_,
                                              grammar_cache=grammar_cache,
                                              max_seq_length=max_seq_length,
                                              decay=decay,
                                              updates_to_refresh=updates_to_refresh)
        elif 'model' in kind:
            my_globals = GlobalParametersModel(batch_size=batch_size,
                                               reward_fun_=reward_fun_,
                                               grammar_cache=grammar_cache,  # 'hyper_grammar.pickle'
                                               max_depth=max_seq_length,
                                               lr=lr,
                                               grad_clip=grad_clip,
                                               entropy_weight=entropy_weight,
                                               decay=decay,
                                               num_bins=num_bins,
                                               updates_to_refresh=updates_to_refresh
                                               )

        with gzip.open(globals_name, 'wb') as f:
            dill.dump(my_globals, f)

    node_type = class_from_kind(kind)
    from generative_playground.utils.deep_getsizeof import memory_by_type
    with Shelve(db_path, 'kv_table', compress=compress_data_store) as state_store:
        my_globals.state_store = state_store
        root_node = node_type(my_globals,
                              parent=None,
                              source_action=None,
                              depth=1)

        for _ in range(num_batches):
            mem = memory_by_type()
            print("memory pre-explore", sum([x[2] for x in mem]), mem[:5])
            rewards, infos = explore(root_node, batch_size)
            state_store.flush()
            mem = memory_by_type()
            print("memory post-explore", sum([x[2] for x in mem]), mem[:5])
            with gzip.open(globals_name, 'wb') as f:
                my_globals.state_store = None
                dill.dump(my_globals, f)
                my_globals.state_store = state_store

            # visualisation code goes here
            plotter_input = {'rewards': np.array(rewards),
                             'info': [[x['smiles'] for x in infos], np.ones(len(infos))]}
            plotter(None, None, plotter_input, None, None)
            print(max(rewards))

    print(root_node.result_repo.avg_reward())
    # temp_dir.cleanup()
    print("done!")

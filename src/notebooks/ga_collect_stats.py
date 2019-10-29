import os, glob, pickle, sys
import torch
import networkx as nx
import gzip

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')

from generative_playground.models.problem.genetic.genetic_opt import populate_data_cache, load_coeff_vector_cache
snapshot_dir = os.path.realpath('../generative_playground/molecules/train/genetic/data')
root_name = 'genetic2_v2_lr0.01_ew0.1'#'AA2scan8_v2_lr0.1_ew0.1' #
dirs = glob.glob(snapshot_dir + '/' + root_name + '*')
lineage_files = []
for dir in dirs:
    lineage_files += glob.glob(dir + '/*lineage.pkl')

graph = nx.DiGraph()
for file in lineage_files:
    with open(file, 'rb') as f:
        new_graph = pickle.load(f)
        graph = nx.compose(graph, new_graph)
        print(len(graph), file)


nx.write_gexf(graph, snapshot_dir + '/' + root_name + ".gexf", version="1.2draft")
# next step: pull the rewards vectors and the model coeffs into the graph
data_cache = {}
coeff_cache = {}
for dir in dirs:
    data_cache = populate_data_cache(dir, data_cache)
    coeff_cache = load_coeff_vector_cache(dir, coeff_cache)

for node_id in graph.nodes:
    if node_id in data_cache:
        print('rewards found for', node_id)
        graph.node[node_id].update(data_cache[node_id])
    else:
        print('NO rewards found for', node_id)

    if node_id in coeff_cache:
        print('coeffs found for', node_id)
        graph.node[node_id].update(coeff_cache[node_id])
    else:
        print('NO coeffs found for', node_id)

with gzip.open(snapshot_dir + '/' + root_name + "_graph.zip", 'wb') as f:
    pickle.dump(graph, f)
print('done!')
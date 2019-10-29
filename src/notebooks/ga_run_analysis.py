import os, glob, pickle, sys
import torch
import networkx as nx

if '/home/ubuntu/shared/GitHub' in sys.path:
    sys.path.remove('/home/ubuntu/shared/GitHub')

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

print('done!')
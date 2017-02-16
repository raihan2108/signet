import networkx as nx
import os
import math
import cPickle as pickle
import operator
import numpy as np
import random
from py_signet import py_signet
from sklearn import linear_model
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split


data_dir = 'data'
# dataset_name = 'scotus'
dataset_name = 'adj_network'
if dataset_name == 'scotus':
    n_vertices = 28305
    label_file = 'scotus_label.txt'
elif dataset_name == 'adj_network':
    n_vertices = 4579
    label_file = 'adj_label.txt'

threshold = 5


def sample_nodes(walks, network):
    samp = dict()
    samn = dict()
    for node in network.nodes():
        samp[node] = []
        samn[node] = []
    for i, node in enumerate(walks):
        if i != 0 and i % 10000 == 0:
            print("done for " + str(i) + " th node")

        for walk in walks[node]:
            # walk = walks[node]
            if len(walk) == 0: continue
            if network.has_edge(node, walk[0]):
                curr_sign = network[node][walk[0]]['weight']
                if curr_sign == 0:
                    curr_sign = -1
            else:
                curr_sign = -1

            for i in xrange(1, len(walk) - 1):
                src = walk[i]
                dest = walk[i + 1]
                if not network.has_edge(src, dest):
                    sign = -1
                else:
                    sign = network[src][dest]['weight']
                    if sign == 0:
                        sign = -1
                curr_sign *= sign
                if curr_sign > 0 and len(samp[node]) < threshold:
                    samp[node].append(dest)
                elif curr_sign < 0 and len(samn[node]) < threshold:
                    samn[node].append(dest)
                elif curr_sign == 0:
                    print('error invalid value ' + str(sign) + ' ' + str(curr_sign))

    print('start removing conflicting nodes')

    for i, node in enumerate(walks):
        if i != 0 and i % 10000 == 0:
            print("done for " + str(i) + " th node")

        conflicting = list(set(samp[node]).intersection(set(samn[node])))
        for c_item in conflicting:
            if nx.has_path(network, source=node, target=c_item):
                all_paths = {}
                for t_path in nx.all_shortest_paths(network, source=node, target=c_item):
                    all_paths[t_path] = 0
                for path in all_paths:
                    c = 0
                    for i in xrange(0, len(path) - 1):
                        curr_sign = network[path[i]][path[i + 1]]['weight']
                        if curr_sign > 0:
                            c += 1
                    all_paths[path] = c
                path = max(all_paths.iteritems(), key=operator.itemgetter(1))[0]
                sign = 1
                for i in xrange(0, len(path) - 1):
                    curr_sign = network[path[i]][path[i + 1]]['weight']
                    if curr_sign == 0:
                        curr_sign = -1
                    sign *= curr_sign
                if sign > 0:
                    samn[node].remove(c_item)
                elif sign < 0:
                    samp[node].remove(c_item)
            else:
                samp[node].remove(c_item)
    return samp, samn


def construct_network(edges, all_vertices, train_nodes):
    signed_network = nx.DiGraph()
    unsigned_network = nx.Graph()
    count = 0
    for v in all_vertices:
        signed_network.add_node(v)
        unsigned_network.add_node(v)
    for i, edge in enumerate(edges):
        source = edge[0]
        destination = edge[1]
        sign = edge[2]

        if source in train_nodes:

            if unsigned_network.has_edge(source, destination):
                unsigned_network[source][destination]['weight'] += sign
            else:
                unsigned_network.add_edge(source, destination, weight=sign)
            if signed_network.has_edge(source, destination):
                signed_network[source][destination]['weight'] += sign
                if signed_network[source][destination]['weight'] == 0:
                    count += 1
                    print'error'
            else:
                signed_network.add_edge(source, destination, weight=sign)
    return signed_network, unsigned_network


def get_edges(dataset_name):
    data_path = os.path.join(data_dir, dataset_name)
    network_name = dataset_name.split('.')[0]
    edges = []
    with open(data_path, 'r') as raw_file:
        for i, line in enumerate(raw_file):
            if line.startswith('#'): continue
            source, destination, sign = map(int, line.split())
            if sign == 0:
                sign = -1
            edges.append((source, destination, sign))
    return edges


def weighted_pick(d):
    r = random.uniform(0, sum(d.itervalues()))
    s = 0.0
    # print r, sum(d.itervalues())
    for k, w in d.iteritems():
        s += w
        if r <= s: return k

    return -1


def random_walk(graph, start_node, size):
    ret = []
    try:
        next_node = random.choice(graph.neighbors(start_node))
    except IndexError:
        return ret
    ret.append(next_node)

    for i in xrange(1, size + 1):
        weights = {}
        for nei in graph.neighbors(next_node):
            if nei not in ret and nei != start_node:
                weights[nei] = abs(graph[next_node][nei]['weight'])
        if len(weights) == 0:
            return ret
        else:
            selected_node = weighted_pick(weights)
        if selected_node == -1:
            return ret
        '''if selected_node not in ret:'''
        ret.append(selected_node)
        next_node = selected_node

    return ret


def get_labels():
    if dataset_name == 'scotus':
        data_path = os.path.join(data_dir, label_file)
        issue_label = np.zeros(n_vertices)
        pol_label = np.zeros(n_vertices)
        with open(data_path) as l_file:
            for line in l_file:
                node, pol, issue = map(int, line.split())
                pol_label[node] = pol
                issue_label[node] = issue

        return pol_label, issue_label
    elif dataset_name == 'adj_network':
        data_path = os.path.join(data_dir, label_file)
        adj_label = np.zeros(n_vertices)
        with open(data_path) as l_file:
            for i, line in enumerate(l_file):
                if line.strip() == 'p':
                    adj_label[i] = 1
                elif line.strip() == 'n':
                    adj_label[i] = 2
        return adj_label


if __name__ == '__main__':
    edges = get_edges(dataset_name + '.txt')
    if dataset_name == 'scotus':
        y_pol, y_issue = get_labels()
    elif dataset_name == 'adj_network':
        y_pol = get_labels()

    test_size = 0.5
    print 'data loaded'
    all_vertices = np.arange(0, n_vertices).tolist()
    train_nodes, test_nodes = train_test_split(all_vertices, test_size=test_size, random_state=random.randint(10, 100))
    signed_network, unsigned_network = construct_network(edges, all_vertices, all_vertices)
    # print 'graph is ' + str(nx.is_connected(unsigned_network))

    print 'network constructed'

    nodes_by_degree_centrality = nx.degree_centrality(signed_network)
    sorted_by_degree_centrality = sorted(nodes_by_degree_centrality.items(), key=operator.itemgetter(1), reverse=True)

    total_walks = n_vertices
    walks = {}
    for i in xrange(0, total_walks):
        current_node = sorted_by_degree_centrality[i][0]
        walks[current_node] = []
        for n in xrange(0, 1):
            walk = random_walk(signed_network, start_node=current_node, size=5)
            walks[current_node].append(walk)

        if i != 0 and i % 10000 == 0:
            print("done for " + str(i) + " th node")

    samp, samn = sample_nodes(walks, signed_network)
    print 'node sampling completed'

    n_edges = signed_network.number_of_edges()
    edge_source = np.zeros(n_edges, dtype=np.int)
    edge_target = np.zeros(n_edges, dtype=np.int)
    edge_weight = np.zeros(n_edges, dtype=np.int)

    i = 0
    for uu, vv, ww in signed_network.edges(data=True):
        edge_source[i] = uu
        edge_target[i] = vv
        edge_weight[i] = ww['weight']
        i += 1
    total_samples = 100
    n_dims = 20
    n_iterations = 10
    n_negatives = 5
    init_rho = 0.025
    order = 2
    is_neg_sampling = False

    node_embeddings = np.zeros(n_vertices * n_dims, dtype=np.float)
    context_embeddings = np.zeros(n_vertices * n_dims, dtype=np.float)
    print('transferring control for executing signet')

    py_signet(edge_source, edge_target, edge_weight, node_embeddings, context_embeddings,
             samp, samn, n_vertices, n_edges, n_dims, init_rho, n_iterations, n_negatives,
             order, total_samples, is_neg_sampling)

    node_emb = np.reshape(node_embeddings, (n_vertices, n_dims))
    context_emb = np.reshape(context_embeddings, (n_vertices, n_dims))

    if order == 2:
        final_emb = np.zeros((n_vertices, 2 * n_dims), dtype=np.float)
        for i in xrange(0, n_vertices):
            for j in xrange(0, n_dims):
                final_emb[i][j] = node_emb[i][j]
                final_emb[i][j + n_dims] = context_emb[i][j]
    elif order == 1:
        final_emb = np.zeros((n_vertices, n_dims), dtype=np.float)
        for i in xrange(0, n_vertices):
            for j in xrange(0, n_dims):
                final_emb[i][j] = node_emb[i][j]

    n_dims = final_emb.shape[1]
    n_features = n_dims

    n_trains = len(train_nodes)
    n_tests = len(test_nodes)

    X_train = []
    y_train_pol = []

    X_test = []
    y_test_pol = []

    print 'constructing examples from embedding'

    for i in xrange(0, n_trains):
        uu = train_nodes[i]
        if y_pol[uu] == 3:
            continue
        X_train.append(final_emb[uu, :].tolist())
        y_train_pol.append(y_pol[uu])

    for i in xrange(0, n_tests):
        uu = test_nodes[i]
        if y_pol[uu] == 3:
            continue
        X_test.append(final_emb[uu, :].tolist())
        y_test_pol.append(y_pol[uu])

    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train_pol = np.asarray(y_train_pol)
    y_test_pol = np.asarray(y_test_pol)
    print X_train.shape, X_test.shape, y_train_pol.shape, y_test_pol.shape

    clf_pol = linear_model.LogisticRegression()
    clf_pol.fit(X_train, y_train_pol)
    y_pred_pol = clf_pol.predict(X_test)
    # y_proba_pol = clf_pol.decision_function(X_test)
    print 'dataset ' + str(dataset_name) + ' test size ' + str(test_size) + ' Neg Sampling ' + str(is_neg_sampling)
    print 'micro f1 score: %0.4f' % f1_score(y_test_pol, y_pred_pol, average='micro')
    print 'macro f1 score: %0.4f' % f1_score(y_test_pol, y_pred_pol, average='macro')
    # print 'roc auc score: %0.4f' % roc_auc_score(y_test_pol, y_proba_pol, average='micro')


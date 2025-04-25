import pandas as pd

max_rows_num = 500000


def get_net_feature_map():
    feature_list = []
    _map = {}
    with open('./data/swat/n_list.txt', 'r', encoding='utf-8') as f:
        for ann in f.readlines():
            ann = ann.strip('\n')
            ann = ann.split("-")
            feature_list.append(ann[0])
            _map[ann[1]] = ann[0]
    return feature_list, _map


def get_graph_struc(feature_list, _map, feature_map):
    net_train = pd.read_csv("./data/swat/net_train.csv")
    edge_indexes = [[], []]
    for i in range(max_rows_num):
        sent = str(net_train.iloc[[i], [13]].values).rstrip("']]").lstrip("[['")
        rece = str(net_train.iloc[[i], [6]].values).rstrip("']]").lstrip("[['")
        if _map.get(rece) and sent in feature_list:
            rece = _map[rece].lstrip("HMI_")
            sent = sent.lstrip("HMI_")
            rece = feature_map.index(rece)
            sent = feature_map.index(sent)
            edge_indexes[0].append(sent)
            edge_indexes[1].append(rece)
    return edge_indexes

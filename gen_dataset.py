import argparse
import scipy.io as sio
import os
import numpy as np
import networkx as nx
import json
from networkx.readwrite import json_graph
import tarfile
import shutil


def data_generate(data_name, link_num, graph_num, flag):
    dirs = "./data/%s/%s" % (flag, data_name)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    data_path = "./data/%s/%s/%s.txt" % (flag, data_name, data_name)
    if os.path.exists(data_path):
        os.remove(data_path)
    data_file = open(data_path, mode="a+")
    data_file.writelines([str(graph_num), "\n"])  # 1st line: `N` number of graphs;
    # for each block of text:
    for i in range(graph_num):
        # a line contains `n l`
        # `n` is number of nodes in the current graph, and `l` is the graph indices (starting from 0)
        data_file.writelines([str(link_num), "\t", str(i), "\n"])
        # for each block of text:
        for j in range(link_num):
            # `t` is the indices of current node (starting from 0), and `m` is the number of neighbors of current node
            data_file.writelines([str(j), "\t", str(link_num - 1)])
            # following `m` numbers indicate the neighbor indices (starting from 0)
            for k in range(link_num):
                if k != j:
                    data_file.writelines(["\t", str(k)])
            data_file.writelines(["\n"])
    data_file.close()

    # generate labels and coords
    data_path = "./data/%s/%s/label.txt" % (flag, data_name)
    tx_path = "./data/%s/%s/tx.txt" % (flag, data_name)
    ty_path = "./data/%s/%s/ty.txt" % (flag, data_name)
    rx_path = "./data/%s/%s/rx.txt" % (flag, data_name)
    ry_path = "./data/%s/%s/ry.txt" % (flag, data_name)

    matfn = "./mat/dataset_%d_%d.mat" % (graph_num, link_num)
    data = sio.loadmat(matfn)
    channel = data["Channel"]
    graph_label = data["Label"]
    distance = data["Distance"]
    tx = data["Tx"]
    ty = data["Ty"]
    rx = data["Rx"]
    ry = data["Ry"]

    # dquan = data["Distance_quan"]
    np.savetxt(data_path, graph_label, fmt="%d")
    np.savetxt(tx_path, tx, fmt="%.4f")
    np.savetxt(ty_path, ty, fmt="%.4f")
    np.savetxt(rx_path, rx, fmt="%.4f")
    np.savetxt(ry_path, ry, fmt="%.4f")

    # 1: CSI
    for i in range(graph_num):
        sub = np.transpose(channel[:, i].reshape(link_num, link_num))
        data_path = "./data/%s/%s/channel_%d.txt" % (flag, data_name, i)
        if os.path.exists(data_path):
            os.remove(data_path)
        np.savetxt(data_path, sub)

    ##################################################################################
    # # 2: distance quantization
    # for i in range(graph_num):
    #     sub = np.transpose(dquan[:, i].reshape(link_num, link_num))
    #     data_path = "./data/%s/%s/distance_%d.txt" % (flag, data_name, i)
    #     if os.path.exists(data_path):
    #         os.remove(data_path)
    #     np.savetxt(data_path, sub)
    # ##################################################################################
    # 3: Distance
    for i in range(graph_num):
        sub = np.transpose(distance[:, i].reshape(link_num, link_num))
        data_path = "./data/%s/%s/distance_%d.txt" % (flag, data_name, i)
        if os.path.exists(data_path):
            os.remove(data_path)
        np.savetxt(data_path, sub)


def make_graph(g, node_tags, graph_id, dist, channel, labels_total, tx, ty, rx, ry):

    g.graph["num_nodes"] = len(node_tags)
    g.graph["graph_id"] = graph_id

    g.add_nodes_from(
        (
            idx,
            {
                "entity": "transmitter_receiver_pair",
                "transceiver_x": tx[idx, graph_id],
                "transceiver_y": ty[idx, graph_id],
                "receiver_x": rx[idx, graph_id],
                "receiver_y": ry[idx, graph_id],
                # "path_loss": path_loss[:, link_idx].tolist(),
                # "power": 0,
                # "weights": weights[link_idx],
                # "wmmse_power": wmmse_power[link_idx],
                "d2d_distance": dist[idx, idx],
                "d2d_channel": channel[idx, idx],
                "label": labels_total[idx, graph_id],
            },
        )
        for idx in range(len(node_tags))
    )

    x, y = zip(*g.edges())
    num_edges = len(x)
    edge_pairs = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
    edge_pairs[:, 0] = x
    edge_pairs[:, 1] = y

    g.add_edges_from(
        [
            (src, dst, {"interfering_d2d_distance": dist[src, dst], "channel": channel[src, dst]})
            for src, dst in edge_pairs
        ],
    )

    # ? CODE FOR GRAPH VISUALIZATION
    # pos = nx.spring_layout(g)
    # edge_labels = dict(
    #     [
    #         (
    #             (
    #                 u,
    #                 v,
    #             ),
    #             round(d["interfering_d2d_distance"], 2),
    #         )
    #         for u, v, d in g.edges(data=True)
    #     ]
    # )
    # nx.draw(g, pos, with_labels=True,
    # node_color=[
    #         "#a3be8c" if data["label"] == 1 else "#bf616a"
    #         for _, data in g.nodes(data=True)
    #     ],connectionstyle="arc3, rad = 0.1",node_size=700)

    # # node_labels = nx.get_node_attributes(g,'d2d_distance')
    # # node_labels = {u : round(v) for u,v in node_labels.items()}
    # # nx.draw_networkx_labels(g, pos, labels = node_labels, font_size=6)
    # nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=6)
    # plt.show()

    return g


def save_dataset(graphs, flag, compress=False):
    serialized_data = []
    for g in graphs:
        parsed_graph = json_graph.node_link_data(g)
        serialized_data.append(parsed_graph)

    with open(f"data/{flag}/data.json", "w") as json_file:
        json.dump(serialized_data, json_file)

    if compress:
        tar = tarfile.open("./data/%s.tar.gz" % flag, "w:gz")
        tar.add("data.json")
        tar.close()
        os.remove("data.json")


def load_data(dname, flag):

    dirs = "./data/%s/%s/%s.txt" % (flag, dname, dname)
    labels_total = np.loadtxt("./data/%s/%s/label.txt" % (flag, dname))
    tx_total = np.loadtxt("./data/%s/%s/tx.txt" % (flag, dname))
    ty_total = np.loadtxt("./data/%s/%s/ty.txt" % (flag, dname))
    rx_total = np.loadtxt("./data/%s/%s/rx.txt" % (flag, dname))
    ry_total = np.loadtxt("./data/%s/%s/ry.txt" % (flag, dname))

    g_list = []
    label_dict = {}
    node_dict = {}

    with open(dirs, "r") as f:
        # 1st line: `N` number of graphs; then the following `N` blocks describe the graphs
        n_g = int(f.readline().strip())  # get the number of graphs
        # for each graph
        for i in range(n_g):
            row = f.readline().strip().split()
            n, l = [int(w) for w in row]  # `n` is number of nodes in the current graph, and `l` is the graph label
            if l not in label_dict:
                mapped = len(label_dict)
                label_dict[l] = mapped
            g = nx.DiGraph()
            node_tags = []
            n_edges = 0
            # for each node
            dist = np.loadtxt("./data/%s/%s/distance_%d.txt" % (flag, dname, i))
            channel = np.loadtxt("./data/%s/%s/channel_%d.txt" % (flag, dname, i))
            for j in range(n):
                g.add_node(j)
                row = f.readline().strip().split()
                row = [
                    int(w) for w in row
                ]  # `t` is the tag of current node, and `m` is the number of neighbors of current node;
                if not row[0] in node_dict:  # row[0]==t
                    mapped = len(node_dict)
                    node_dict[row[0]] = mapped
                node_tags.append(node_dict[row[0]])
                n_edges += row[1]  # row[1]==m
                for k in range(2, len(row)):
                    g.add_edge(j, int(row[k]))  # following `m` numbers indicate the neighbor indices (starting from 0).
            # assert len(g.edges()) * 2 == n_edges
            assert len(g) == n
            g_new = make_graph(g, node_tags, l, dist, channel, labels_total, tx_total, ty_total, rx_total, ry_total)
            g_list.append(g_new)

    for g in g_list:
        g.graph["graph_id"] = label_dict[g.graph["graph_id"]]

    shutil.rmtree(f"./data/{flag}/{dname}")
    return g_list


# def matlab_generate(train_num, val_num, test_num, d2d):
#     import matlab.engine

#     eng = matlab.engine.start_matlab()
#     eng.addpath("./FPlinQ/")
#     eng.pygenerate(train_num, val_num, test_num, d2d, nargout=0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train", help="Train dataset size", type=int, default=1000)
    parser.add_argument("-v", "--val", help="Validation dataset size", type=int, default=500)
    parser.add_argument("-s", "--test", help="Test dataset size", type=int, default=200)
    parser.add_argument("-d", "--d2d", help="Number of D2D pairs", type=int, default=10)

    dname = "dataset"

    args = parser.parse_args()
    train_num = args.train
    val_num = args.val
    test_num = args.test
    d2d = args.d2d

    # ? Code to generate mat files if not found
    # if not os.path.exists("./mat"):
    #     print("mat folder not found. Creating")
    #     os.makedirs("./mat")
    #     print("generating mat files")
    #     matlab_generate(train_num, val_num, test_num, d2d)
    # if not os.path.exists(f"./mat/dataset_{train_num}_{d2d}.mat"):
    #     print("generating mat files")
    #     matlab_generate(train_num, val_num, test_num, d2d)

    data_generate(data_name=dname, link_num=d2d, graph_num=train_num, flag="train")
    data_generate(data_name=dname, link_num=d2d, graph_num=val_num, flag="val")
    data_generate(data_name=dname, link_num=d2d, graph_num=test_num, flag="test")

    train_graphs = load_data(dname, flag="train")
    val_graphs = load_data(dname, flag="val")
    test_graphs = load_data(dname, flag="test")

    save_dataset(train_graphs, "train")
    save_dataset(val_graphs, "val")
    save_dataset(test_graphs, "test")

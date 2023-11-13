import itertools

import matplotlib.pyplot as plt
import networkx as nx
from brian2 import *


def visualise_connectivity(S):
    """matplotlibを使用してネットワークを可視化します。２層のネットワークのみ対応しています。

    Args:
        S (networkx.Synapsesオブジェクト): シナプス
    """
    Ns = len(S.source)
    Nt = len(S.target)
    figure(figsize=(10, 4))
    subplot(121)
    plot(zeros(Ns), arange(Ns), "ok", ms=10)
    plot(ones(Nt), arange(Nt), "ok", ms=10)
    for i, j in zip(S.i, S.j):
        plot([0, 1], [i, j], "-k")
    xticks([0, 1], ["Source", "Target"])
    ylabel("Neuron index")
    xlim(-0.1, 1.1)
    ylim(-1, max(Ns, Nt))
    subplot(122)
    plot(S.i, S.j, "ok")
    xlim(-1, Ns)
    ylim(-1, Nt)
    xlabel("Source neuron index")
    ylabel("Target neuron index")
    plt.show()


def visualise_network(S: dict):
    """ネットワークの全シナプスを格納した辞書型変数を受け取り，それを元にネットワーク全体の構造を可視化します

    Args:
        S (dict): ネットワークの全シナプスを格納した辞書型変数
        example: G = {"inp": NeuronGroup(5, "v:1"), "exc": NeuronGroup(5, "v:1"), "inh": NeuronGroup(5, "v:1")}
                 S = {"inp2exc": Synapses(G["inp"], G["exc"]), "exc2inh": Synapses(G["exc"], G["inh"])}
    """
    neuron_num = [len(S[key].source) for key in S.keys()]  # 各層のニューロン数を格納
    neuron_num.append(len(list(S.values())[-1].target))  # 最後の層のニューロン数を格納
    # print("neuron_num: ", neuron_num)
    extents = nx.utils.pairwise(itertools.accumulate((0,) + tuple(neuron_num)))
    layers = [range(start, end) for start, end in extents]
    # print("layers:", layers)

    G = nx.Graph()
    for i, layer in enumerate(layers):  # 各層のニューロンをノードとして追加
        G.add_nodes_from(layer, layer=i)
    for layer_num in range(len(layers) - 1):  # 各層間のシナプスをエッジとして追加
        # layer_num : 層数
        # print("layer_num: ", layer_num)
        for j in range(len(S.keys())):
            # j : シナプス数
            synapse = S[list(S.keys())[layer_num]]  # シナプスのリストを取得
            synapse_i = list(synapse.i)
            synapse_j_tmp = list(synapse.j)
            synapse_j = [
                x + neuron_num[layer_num] for x in synapse.j
            ]  # シナプスのターゲットニューロンのインデックスを取得(層のニューロン数を加算)

            # print("synapse_i    : ", synapse_i)
            # print("synapse_j_tmp: ", synapse_j_tmp)

            # ２層目以降は，以前の層の文を加算する
            if layer_num > 0:
                layer_num_sum = sum(neuron_num[:layer_num])
                synapse_i = [x + layer_num_sum for x in synapse_i]
                synapse_j = [x + layer_num_sum for x in synapse_j]

            # print("[Calculated value is below]")
            # print("synapse_i    : ", synapse_i)
            # print("synapse_j    : ", synapse_j)

            synapse_con = []
            for i in range(len(synapse_i)):
                synapse_con.append(
                    [synapse_i[i], synapse_j[i]]
                )  # シナプスのソースニューロンとターゲットニューロンのインデックスをリストにしてリストに格納
            G.add_edges_from(synapse_con)  # シナプスをエッジとして追加
            # print("edge appended: ", synapse_con)

    # for layer1, layer2 in nx.utils.pairwise(layers):
    #     G.add_edges_from(itertools.product(layer1, layer2))
    color = [
        "gold" if data["layer"] % 2 == 0 else "violet" for v, data in G.nodes(data=True)
    ]
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, node_color=color, with_labels=False)
    plt.axis("equal")
    
    
    plt.legend()
    plt.show()

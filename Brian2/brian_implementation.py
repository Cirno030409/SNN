import itertools

import brian_myfunc as my
import matplotlib.pyplot as plt
import networkx as nx
from brian2 import *


def main():
    start_scope()
    G, S, num = {}, {}, {}

    # ニューロン数の定義
    num["inp"] = 10
    num["exc"] = 10
    num["inh"] = 10

    # ニューロングループの定義
    G["inp"] = NeuronGroup(num["inp"], "v:1")
    G["exc"] = NeuronGroup(num["exc"], "v:1")
    G["inh"] = NeuronGroup(num["inh"], "v:1")

    # シナプスの定義
    S["inp2exc"] = Synapses(G["inp"], G["exc"])
    S["exc2inh"] = Synapses(G["exc"], G["inh"])

    # シナプスの接続
    S["inp2exc"].connect(condition="i!=j", p=0.4)
    S["exc2inh"].connect(condition="i!=j", p=0.4)

    my.visualise_network(S)


if __name__ == "__main__":
    main()

import itertools

import brian_myfunc as my
import matplotlib.pyplot as plt
import networkx as nx
from brian2 import *


def main():
    start_scope()
    N, S, num = {}, {}, {}
    # ネットワーク作成
    ## ニューロン数の定義
    num["inp"] = 10
    num["exc"] = 10
    num["inh"] = 10

    ## ニューロンのモデルの定義
    v0 = -50 * mV  # 静止膜電位
    v_th = -40 * mV  # 閾値電位
    v_re = -60 * mV  # リセット電位
    tau = 10 * ms  # 膜時定数
    ref = 2 * ms  # 不応期
    E_ex = 0 * mV  # 興奮シナプスの平衡電位
    E_In = -100 * mV  # 抑制シナプスの平衡電位
    eqs = """
    dv/dt = (v0-v+(I_synE+I_synI)/nS)/tau : volt (unless refractory)
    I_synE = ge * nS * (E_ex -v) : amp
    I_synI = gi * nS * (E_In - v) : amp
    dge/dt = -ge/(1.0*ms) : 1
    dgi/dt = -gi/(2.0*ms) : 1
    """

    ## ニューロングループの定義
    N["inp"] = PoissonGroup(num["inp"], 10 * Hz)  # 入力
    N["exc"] = NeuronGroup(
        num["exc"],
        eqs,
        threshold="v>v_th",
        reset="v=v_re",
        refractory=ref,
        method="euler",
    )  # 興奮性ニューロン
    N["inh"] = NeuronGroup(
        num["inh"],
        eqs,
        threshold="v>v_th",
        reset="v=v_re",
        refractory=ref,
        method="euler",
    )  # 抑制性ニューロン
    N["exc"].v = N["inh"].v = -70 * mV  # 初期膜電位

    ## シナプスの定義
    ### STDP Synapses
    tc_pre_ee = tc_post_ee = 20 * ms  # STDPの時定数
    nu_ee_pre = nu_ee_post = 0.001  # 学習率
    model = """
            w : 1
            dpre/dt   =   -pre/(tc_pre_ee)         : 1 (event-driven)
            dpost/dt  = -post/(tc_post_ee)     : 1 (event-driven)
            """
    pre = """
        pre = 1.
        w = w - nu_ee_pre * post
        ge_post += w
        """
    post = """
        post = 1.
        w = w + nu_ee_post * pre
        """
    weightMatrix = np.random.rand(num["inp"], num["exc"])  # 最初のweightは一様乱数

    S["inp2exc"] = Synapses(N["inp"], N["exc"], model=model, on_pre=pre, on_post=post)
    S["inp2exc"].connect(True)
    S["inp2exc"].w = weightMatrix.flatten()

    ### Stable Synapses
    ei_w = 1.0
    weightMatrix = ei_w * np.eye(num["exc"])
    S["exc2inh"] = Synapses(N["exc"], N["inh"], model="w : 1", on_pre="ge_post += w")
    S["exc2inh"].connect(True)
    S["exc2inh"].w = weightMatrix.flatten()

    ie_w = 1.0 / num["inh"]
    weightMatrix = ie_w * (np.ones(num["exc"]) - np.eye(num["exc"]))
    S["inh2exc"] = Synapses(N["inh"], N["exc"], model="w : 1", on_pre="gi_post += w")
    S["inh2exc"].connect(True)
    S["inh2exc"].w = weightMatrix.flatten()

    ## モニターの定義
    monitors = {{}}
    monitors["rate"]["inp"] = PopulationRateMonitor(N["inp"])
    monitors["rate"]["exc"] = PopulationRateMonitor(N["exc"])
    monitors["rate"]["inh"] = PopulationRateMonitor(N["inh"])

    monitors["spike"]["inp"] = SpikeMonitor(N["inp"])
    monitors["spike"]["exc"] = SpikeMonitor(N["exc"])
    monitors["spike"]["inh"] = SpikeMonitor(N["inh"])

    ## ネットワーク
    net = Network()
    for obj_list in [
        N.values(),
        S.values(),
        monitors["rate"].values(),
        monitors["spike"].values(),
    ]:
        for key in obj_list:
            net.add(obj_list[key])

    net.scheduling_summary()
    my.visualise_network(S)


if __name__ == "__main__":
    main()

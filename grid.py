#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import namedtuple
import random
import numpy as np
from pyscipopt import Model, quicksum
import time

ClusterBid = namedtuple("ClusterBid", ['index', 'energy_cut', 'real_cost'])

class Grid:
    def __init__(self, cluster_count, total_energy_cut):
        self.cluster_count = cluster_count
        self.total_energy_cut = total_energy_cut
        self.cluster_bid_list = []

        # 记录AMEDR算法的相关信息
        self.cur_solution = dict()

        # 记录offline opt的相关信息
        self.best_solution = dict()

    # 使用近似算法生成cur_solution
    def AMEDR_scheduling(self):
        start = time.clock()
        energy_cut_array = np.array([cluster.energy_cut for cluster in self.cluster_bid_list])
        real_cost_array = np.array([cluster.real_cost for cluster in self.cluster_bid_list])
        # remain = delta_S表示剩余需要减少的电量
        remain = self.total_energy_cut
        # schedule以0,1 list形式说明选择那些个cluster
        cur_schedule = [0] * self.cluster_count
        # choose_i表示看dual_variables先达到哪个constrains
        choose_i = -1
        RHS_array = real_cost_array.copy()
        while remain > 0:
            if choose_i >= 0:
                # print("select i={}".format(choose_i))
                cur_schedule[choose_i] = 1
                remain -= energy_cut_array[choose_i]
                # print("after selected, remain={}".format(remain))
                # print("current_set = {}".format(self.cur_solution))
                choose_i = -1
            else:
                delta_s = remain
                Ei_S_array = np.ones(self.cluster_count, dtype=np.float64)
                for i in [index for index, value in enumerate(self.cur_solution) if value == 0]:
                    ei = self.cluster_bid_list[i].energy_cut
                    Ei_S_array[i] = min(ei, delta_s)
                choose_i = (RHS_array / Ei_S_array).argmin()
                multiplier = RHS_array[choose_i] / Ei_S_array[choose_i]
                # 每一步都改变RHS的值，其中一个为0(constrain为tight)，其余为正
                RHS_array = RHS_array - (Ei_S_array * multiplier)
                RHS_array[choose_i] = np.Infinity
        end = time.clock()
        self.cur_solution["execute_time"] = end - start
        self.cur_solution["optimal"] = sum(value * real_cost_array[index] for index, value in enumerate(cur_schedule))
        self.cur_solution["accepted_rate"] = sum(cur_schedule) / self.cluster_count
        self.cur_solution["schedule"] = cur_schedule

    # 得到model最优解
    def get_best(self):
        # 使用MIP解决问题 我感觉这里用的是simplex algorithm？？？
        start = time.clock()
        model = Model("primal-dual problem")
        best_schedule = [0] * self.cluster_count

        decision_variables = {}
        for i in range(self.cluster_count):
            decision_variables[i] = model.addVar(vtype='B', name='Z[%d]' % i)
        model.addCons(cons=(quicksum(decision_variables[i] * self.cluster_bid_list[i].energy_cut
                                     for i in range(self.cluster_count))
                            >= self.total_energy_cut),
                      name='satisfy energy demand response')
        model.setObjective(coeffs=quicksum(decision_variables[i] * self.cluster_bid_list[i].real_cost
                                           for i in range(self.cluster_count)),
                           sense='minimize')
        model.optimize()

        end = time.clock()
        self.best_solution["execute_time"] = end - start
        self.best_solution["is_feasible"] = model.getStatus() == "optimal"
        self.best_solution["optimal"] = model.getObjVal()
        for i in range(self.cluster_count):
            if int(round(model.getVal(decision_variables[i]))) == 1:
                best_schedule[i] = 1
        self.best_solution["accepted_rate"] = sum(best_schedule) / self.cluster_count
        self.best_solution["schedule"] = best_schedule

    # 随机生成cluster的出价bid
    def generate_cluster_bid_list(self, energy_cut_mid=150):
        # 标准cluster耗电600kwh
        # 削减10%=60kwh; 削减15%=90kwh; 削减20%=120kwh; 削减25%=150kwh

        # 正常工业用电，大约是0.1$/kwh
        electricity_price = 0.1
        result_list = []
        for i in range(self.cluster_count):
            # 在标准削减电量的情况下存在0.8-1.2的浮动
            cur_energy_cut = energy_cut_mid * random.uniform(0.8, 1.2)
            # EDR情况下的价格，是正常价格的11-18倍
            energy_price = electricity_price * random.uniform(11.0, 18.0)
            cur_cluster = ClusterBid(index=i, energy_cut=cur_energy_cut,
                                     real_cost=cur_energy_cut*energy_price)
            result_list.append(cur_cluster)
        return result_list

    def print_cluster_list(self):
        output = str()
        for cluster in self.cluster_bid_list:
            output += 'cluster[%d]:energy_cut=%f, real_cost=%f\n' % (cluster.index, cluster.energy_cut, cluster.real_cost)
        print(output)

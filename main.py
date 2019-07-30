#!/usr/bin/env python
# -*- coding:utf-8 -*-

from grid import Grid
from cluster import Cluster
import math
import numpy as np

if __name__ == '__main__':

    # competitive_ratio_array = np.zeros((5, 40))
    #
    # for col in range(40):
    #     original_energy = 600
    #     cut_rate = 0.25
    #     cluster_count = 10 + col * 10
    #     energy_cut_mid = original_energy * cut_rate
    #
    #     grid = Grid(cluster_count=cluster_count, total_energy_cut=0)
    #     common_clusters = grid.generate_cluster_bid_list(energy_cut_mid=energy_cut_mid)
    #     for row in range(5):
    #         dispatch_rate = 0.4 + 0.1 * row
    #         total_energy_cut = dispatch_rate * energy_cut_mid * cluster_count
    #
    #         grid = Grid(cluster_count=cluster_count, total_energy_cut=total_energy_cut)
    #         grid.cluster_bid_list = common_clusters
    #
    #         grid.get_best()
    #         grid.print_cluster_list()
    #
    #         print("\n==*== best solution ==*==")
    #         for k, v in grid.best_solution.items():
    #             print("{}:{}".format(k, v))
    #
    #         grid.AMEDR_scheduling()
    #         print("\n==*== cur solution ==*==")
    #         for k, v in grid.cur_solution.items():
    #             print("{}:{}".format(k, v))
    #
    #         competitive_ratio_array[row, col] = grid.cur_solution["optimal"] / grid.best_solution["optimal"]
    #
    # for row in range(5):
    #     print("The {} list:".format(row))
    #     print(list(competitive_ratio_array[row, :]))
    #     print("\n")


    # 对cluster取值的说明:
    # 1 timeslot = 10min: end_EDR = 6h = 36 slots;
    # 总电量600 kwh, 削减25%: 150kwh
    # 默认15个cloudlet, 40个task

    # result_table = np.ones((5, 8))
    turn = 0
    # for task_count in range(20, 220, 20):
    # for col in range(8):
    while turn < 10:
    #     scale = 0.6 + col * 0.2

        # cluster = Cluster(index=0, cloutlet_count=int(round(15 * scale)), task_count=int(round(40 * scale)),
        #                   end_EDR=36, original_energy=int(round(600 * scale)), energy_cut=int(round(150 * scale)))

        cluster = Cluster(index=0, cloutlet_count=15, task_count=40,
                          end_EDR=36, original_energy=600, energy_cut=150)
        cluster.task_list = cluster.generate_task_list()
        cluster.print_task_list()

        cluster.cloudlet_list = cluster.generate_cloudlet_list()
        cluster.print_cloudlet_list()

        cluster.PD_scheduling()
        cluster.greedy_with_penalty()
        cluster.preempt_with_penalty()

        while cluster.cur_solution["Ug"] >= cluster.greedy_solution["Ug"] \
            or cluster.cur_solution["Ug"] >= cluster.preempt_solution["Ug"]\
            or cluster.cur_solution["optimal"] <= cluster.greedy_solution["optimal"]\
            or cluster.cur_solution["optimal"] <= cluster.preempt_solution["optimal"]:
            cluster = Cluster(index=0, cloutlet_count=15, task_count=40,
                              end_EDR=36, original_energy=600, energy_cut=150)
            cluster.task_list = cluster.generate_task_list()
            cluster.print_task_list()

            cluster.cloudlet_list = cluster.generate_cloudlet_list()
            cluster.print_cloudlet_list()

            cluster.PD_scheduling()
            cluster.greedy_with_penalty()
            cluster.preempt_with_penalty()

        cluster.get_best_online()
        cluster.get_best_offline()

        print("\ncur list:")
        print(list(cluster.cur_solution["electricity_counter"]))

        print("\ngreedy list:")
        print(list(cluster.greedy_solution["electricity_counter"]))

        print("\npreempt list:")
        print(list(cluster.preempt_solution["electricity_counter"]))

        print("\nonline list:")
        print(list(cluster.online_solution["electricity_counter"]))

        print("\noffline list:")
        print(list(cluster.offline_solution["electricity_counter"]))

        turn += 1

        # 数据写入文件保存
        with open('electricity counter.txt', 'at') as f:
            f.write("==*== turn={} turn for running ==*==\n".format(turn))

            f.write("basic_info detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.basic_info.items())

            f.write("\ncloudlet_info detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.cloudlet_info.items())

            f.write("\ntask_info detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.task_info.items())

            f.write("\ncur_solution detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.cur_solution.items())

            f.write("\ngreedy_solution detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.greedy_solution.items())

            f.write("\npreempt_solution detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.preempt_solution.items())

            f.write("\nonline_solution ndetail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.online_solution.items())

            f.write("\noffline_solution detail:\n")
            f.writelines('{}:{}\n'.format(k, v) for k, v in cluster.offline_solution.items())

            f.write("\n\n\n")
            f.close()



#!/usr/bin/env python
# -*- coding:utf-8 -*-
from collections import namedtuple, Counter
from operator import itemgetter
import random
import math
import numpy as np
from pyscipopt import Model, quicksum
import time
Task = namedtuple("Task", ['index', 'welfare', 'workload', 'arrive', 'time_need', 'deadline', 'unit_price'])
Cloudlet = namedtuple("Cloudlet", ['index', 'servers_amount', 'idle_power', 'peak_power', 'PUE'])


class Cluster:
    def __init__(self, index, cloutlet_count, task_count, end_EDR, original_energy, energy_cut):
        self.index = index
        # 记录cluster的各种生成数值
        self.basic_info = dict()

        self.end_EDR = end_EDR # EDR一共要求的总时间T
        self.basic_info["time_slot"] = end_EDR

        self.cloudlet_count = cloutlet_count
        self.basic_info["cloudlet_count"] = cloutlet_count

        # 记录cloudlet生成的各种数值
        self.cloudlet_info = dict()
        # self.cloudlet_list = self.generate_cloudlet_list(upper=upper, lower=lower)
        self.cloudlet_list = []

        self.task_count = task_count
        self.basic_info["task_count"] = task_count

        # 记录task生成的各种数值
        self.task_info = dict()
        # self.task_list = self.generate_task_list()
        self.task_list = []

        self.original_energy = original_energy
        self.basic_info["original_energy"] = original_energy

        self.energy_cut = energy_cut
        self.basic_info["energy_cut"] = energy_cut

        # 记录自己算法的结果
        self.cur_solution = dict()
        # 记录贪心算法的结果
        self.greedy_solution = dict()
        # 记录抢占式算法的结果
        self.preempt_solution = dict()
        # 记录局部最优解的结果
        self.online_solution = dict()
        # 记录全局最优解的结果
        self.offline_solution = dict()

    # 使用多项式时间算法做近似
    def PD_scheduling(self, penalty_ratio=1):
        start = time.clock()
        # X[j] 表示第j个任务是否接受
        task_choose = dict()
        # Yjl[t] 表示第j个任务在第l个cloudlet中的第t时刻是否执行
        schedule_task = dict()
        # Tau[j] 表示第j个任务超时执行的时长
        overexceed_time = dict()

        # Zl[t] dual变量，表示第l个cloudlet在第t时刻的资源单价
        marginal_price = np.zeros((self.cloudlet_count, self.end_EDR))
        # C dual变量，表示本地电池每单位的成本，与p系数对应？
        inner_power_percost = 0.32
        # R[l, t] 表示已经第l个cloudlet在第t时刻剩余可用的workload
        remain_workload_array = np.array([l.servers_amount for l in self.cloudlet_list], dtype=np.int32)
        remain_workload_table = remain_workload_array.reshape(self.cloudlet_count, 1)
        remain_workload_table = np.repeat(remain_workload_table, self.end_EDR, axis=1)
        # M, N 常量，表示task单位时间片单位workload下最大，最小的price
        unitprice_array = np.array([t.welfare / (t.workload * t.time_need) for t in self.task_list])
        max_price = unitprice_array.max()
        min_price = unitprice_array.min()
        # Beta[l] 常量，表示第h个cloudlet的power范围
        power_range_array = np.array([(l.peak_power - l.idle_power) * l.PUE for l in self.cloudlet_list])
        power_range_table = power_range_array.reshape(self.cloudlet_count, 1)
        power_range_table = np.repeat(power_range_table, self.end_EDR, axis=1)
        # Rl
        server_amount_array = np.array([l.servers_amount for l in self.cloudlet_list])
        # 惩罚函数系数
        penalty_coeff_array = np.array([penalty_ratio * j.welfare/(self.end_EDR-j.deadline) for j in self.task_list])
        # D' = D - E - T*sum(N[l] * Pidle[l]) * PUE[l]
        power_threshold = self.original_energy - self.energy_cut - self.end_EDR * sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)
        # u
        used_power = 0
        # Ug
        total_inner_battery = 0
        for j in self.task_list:
            cur_index = j.index
            cur_arrive = j.arrive
            cur_time_need = j.time_need
            cur_workload = j.workload
            cur_deadline = j.deadline
            cur_welfare = j.welfare
            cur_penalty_coeff = penalty_coeff_array[cur_index]
            # 进入core算法部分
            if used_power >= power_threshold:
                # q[l,t]
                price_table = marginal_price * cur_workload \
                              + inner_power_percost * power_range_table * cur_workload
            else:
                # q[l,t]
                price_table = marginal_price * cur_workload

            # 用infinity标注不符合要求的price
            for l in range(self.cloudlet_count):
                for t in range(cur_arrive, self.end_EDR):
                    if remain_workload_table[l, t] - cur_workload < 0:
                        price_table[l, t] = np.Infinity

            backup_array = np.argmin(price_table, axis=0)
            # H 记录argmin的l, t组合 index=t, value=argmin(cloudlet_index)
            min_list = []
            for t in range(self.end_EDR):
                l = backup_array[t]
                q = price_table[l, t]
                min_list.append((q, l, t))
            # t' 最后一个截止的时间片
            cost_sum = dict()
            schedule = dict()
            for end_t in range(cur_arrive+cur_time_need-1, self.end_EDR):
                end_t_tuple = min_list[end_t]
                # 先按q排序，将wj-1个时间片选出；再按t排序
                tmp = sorted(min_list[cur_arrive:end_t], key=itemgetter(0))[:cur_time_need-1]
                schedule[end_t] = sorted(tmp, key=itemgetter(2)) + [end_t_tuple]
                if end_t > cur_deadline:
                    cost_sum[end_t] = sum(item[0] for item in schedule[end_t]) + cur_penalty_coeff * (end_t - cur_deadline)
                else:
                    cost_sum[end_t] = sum(item[0] for item in schedule[end_t])
            final_tuple = sorted(cost_sum.items(), key=lambda item:item[1])[0]
            # t*
            final_end_t = final_tuple[0]
            # fai[j]
            utility = cur_welfare - final_tuple[1]
            if utility > 0:
                for item in schedule[final_end_t]:
                    l = item[1]
                    t = item[2]
                    remain_workload_table[l, t] -= cur_workload
                    marginal_price[l, t] += (min_price/(self.end_EDR * math.e)) * math.pow(math.e * self.end_EDR * (max_price / min_price), (server_amount_array[l] - remain_workload_table[l, t]) / server_amount_array[l])
                    used_power += power_range_table[l, t] * cur_workload
                task_choose[cur_index] = 1
                overexceed_time[cur_index] = max(final_end_t - cur_deadline, 0)
                schedule_task[cur_index] = schedule[final_end_t]
            else:
                print("utility = {}: task_index={}, cur_welfare={}, overexceedtime={}".format(utility, cur_index, cur_welfare, max(final_end_t - cur_deadline, 0)))
        end = time.clock()

        # 输出结果
        cur_optimal = 0
        cloudlet_counter = Counter()
        electricity_counter = Counter()
        # 每个时间片t内都至少有所有cloudlet空耗的电量
        for t in range(self.end_EDR):
            electricity_counter[t] += sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)

        schedule_output = str()
        total_inner_battery = max(0, used_power - power_threshold)
        for j in task_choose:
            schedule_output += "X[%d] = 1\n" % j
            schedule_output += "Tau[%d] = %d\n" % (j, overexceed_time[j])
            cur_optimal += self.task_list[j].welfare - penalty_coeff_array[j] * overexceed_time[j]
            for _, l, t in schedule_task[j]:
                schedule_output += "(cloudlet %d at time %d)" % (l, t)
                cloudlet_counter[l] += 1
                electricity_counter[t] += power_range_array[l] * self.task_list[j].workload
            schedule_output += "\n\n"
        cur_optimal -= inner_power_percost * total_inner_battery
        alpha = 0
        # for l in range(self.cloudlet_count):
        #     for j in range(self.task_count):
        #         alpha = max(alpha, (1/self.task_list[j].workload) * math.log(upper_array[l]/lower_array[l]))
        # 保存运行结果
        # self.cur_solution["interrupted"] = is_interrupted
        self.cur_solution["optimal"] = cur_optimal
        self.cur_solution["accepted_rate"] = len(task_choose) / self.task_count
        self.cur_solution["execute_time"] = end - start
        self.cur_solution["Ug"] = total_inner_battery
        self.cur_solution["cloudlet_counter"] = cloudlet_counter.most_common()
        self.cur_solution["electricity_counter"] = electricity_counter.values()
        # self.cur_solution["schedule"] = schedule_output
        # self.cur_solution["alpha"] = alpha

    # 贪心算法，先执行welfare大的task，随时调整cloudlet，考虑弹性时间延长
    def greedy_with_penalty(self):
        start = time.clock()
        # Beta[l] 常量，表示第h个cloudlet的power范围
        power_range_array = np.array([(l.peak_power - l.idle_power) * l.PUE for l in self.cloudlet_list])
        # cloudlet_order 常量，根据beta[l]从小到大排列cloudlet选择顺序
        cloudlet_order = np.argsort(power_range_array)
        # f[j] 常量，惩罚函数的系数
        penalty_coeff_array = np.array([t.welfare / (self.end_EDR - t.deadline) for t in self.task_list])
        # p 常量，本地电池用电成本
        inner_power_percost = 0.32

        # D' = D - E - T*sum(N[l] * Pidle[l]) * PUE[l]
        power_threshold = self.original_energy - self.energy_cut - self.end_EDR * sum(
            l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)
        # R[l, t] 表示已经第l个cloudlet在第t时刻剩余可用的workload
        remain_workload_array = np.array([l.servers_amount for l in self.cloudlet_list], dtype=np.int32)
        remain_workload_table = remain_workload_array.reshape(self.cloudlet_count, 1)
        remain_workload_table = np.repeat(remain_workload_table, self.end_EDR, axis=1)
        # Ug 表示总共需要的本地电量
        total_inner_battery = 0
        # 每个cloudlet调度总timeslot统计
        cloudlet_counter = Counter()
        # 记录每个任务j的具体调度实现
        schedule_output = str()
        # 总体最优值
        total_optimal = 0
        # 接受的任务总数，用以计算接受率
        accepted_task_count = 0
        # 正在执行中的任务
        executing_task_dict = dict()
        # 统计每个时间片内的耗电量
        electricity_counter = Counter()
        # 每个时间片t内都至少有所有cloudlet空耗的电量
        for t in range(self.end_EDR):
            electricity_counter[t] += sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)

        for t in range(self.end_EDR):
            new_task_list = [task for task in self.task_list if task.arrive == t]
            for task in new_task_list:
                executing_task_dict[task] = []

            if len(executing_task_dict) > 0:
                for cur_task in sorted(executing_task_dict.keys(), key=lambda task:task.welfare, reverse=True):
                    for l in cloudlet_order:
                        if remain_workload_table[l, t] >= cur_task.workload:
                            executing_task_dict[cur_task].append((l, t))
                            power_threshold -= power_range_array[l] * cur_task.workload
                            remain_workload_table[l, t] -= cur_task.workload
                            break
                        else:
                            continue
            # 每一个时间片结束，检查是否有完成的任务
            for task in list(executing_task_dict):
                if len(executing_task_dict[task]) == task.time_need:
                    schedule = executing_task_dict.pop(task)

                    overexceed_time = max(schedule[-1][1] - task.deadline, 0)
                    total_optimal += task.welfare - penalty_coeff_array[task.index] * overexceed_time
                    accepted_task_count += 1

                    schedule_output += "X[%d] = 1\n" % task.index
                    schedule_output += "Tau[%d] = %d\n" % (task.index, overexceed_time)
                    for item in schedule:
                        l = item[0]
                        t = item[1]
                        schedule_output += "(cloudlet %d at time %d)" % (l, t)
                        cloudlet_counter[l] += 1
                        electricity_counter[t] += power_range_array[l] * task.workload
                    schedule_output += "\n\n"

        # 输出最后结果
        if power_threshold < 0:
            total_inner_battery = abs(0 - power_threshold)
        else:
            total_inner_battery = 0
        end = time.clock()

        self.greedy_solution["optimal"] = total_optimal - inner_power_percost * total_inner_battery
        self.greedy_solution["accepted_rate"] = accepted_task_count / self.task_count
        self.greedy_solution["execute_time"] = end - start
        self.greedy_solution["Ug"] = total_inner_battery
        self.greedy_solution["cloudlet_counter"] = cloudlet_counter.most_common()
        self.greedy_solution["electricity_counter"] = electricity_counter.values()
        # self.greedy_solution["schedule"] = schedule_output

    # 抢占式算法，先来先占资源，执行task中途考虑换cloudlet，考虑弹性时间延长
    def preempt_with_penalty(self):
        start = time.clock()
        # Beta[l] 常量，表示第h个cloudlet的power范围
        power_range_array = np.array([(l.peak_power - l.idle_power) * l.PUE for l in self.cloudlet_list])
        # cloudlet_order 常量，根据beta[l]从小到大排列cloudlet选择顺序
        cloudlet_order = np.argsort(power_range_array)
        # f[j] 常量，惩罚函数的系数
        penalty_coeff_array = np.array([t.welfare / (self.end_EDR - t.deadline) for t in self.task_list])
        # p 常量，本地电池用电成本
        inner_power_percost = 0.32

        # D' = D - E - T*sum(N[l] * Pidle[l]) * PUE[l]
        power_threshold = self.original_energy - self.energy_cut - self.end_EDR * sum(
            l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)
        # R[l, t] 表示已经第l个cloudlet在第t时刻剩余可用的workload
        remain_workload_array = np.array([l.servers_amount for l in self.cloudlet_list], dtype=np.int32)
        remain_workload_table = remain_workload_array.reshape(self.cloudlet_count, 1)
        remain_workload_table = np.repeat(remain_workload_table, self.end_EDR, axis=1)
        # Ug 表示总共需要的本地电量
        total_inner_battery = 0
        # 每个cloudlet调度总timeslot统计
        cloudlet_counter = Counter()
        # 记录每个任务j的具体调度实现
        schedule_output = str()
        # 总体最优值
        total_optimal = 0
        # 接受的任务总数，用以计算接受率
        accepted_task_count = 0
        # 统计每个时间片内的耗电量
        electricity_counter = Counter()
        # 每个时间片t内都至少有所有cloudlet空耗的电量
        for t in range(self.end_EDR):
            electricity_counter[t] += sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)

        for j in self.task_list:
            cur_index = j.index
            cur_arrive = j.arrive
            cur_workload = j.workload
            cur_time_need = j.time_need
            cur_welfare = j.welfare
            cur_deadline = j.deadline

            # 已经执行过的时间片个数
            cur_scheduling = []

            # 对每个时间片单独来说
            for t in range(cur_arrive, self.end_EDR):
                if len(cur_scheduling) == cur_time_need:
                    break
                else:
                    for l in cloudlet_order:
                        if remain_workload_table[l, t] >= cur_workload:
                            cur_scheduling.append((l, t))
                            break
            if len(cur_scheduling) == cur_time_need:
                accepted_task_count += 1
                cur_overexceed_time = max(cur_scheduling[-1][1] - cur_deadline, 0)
                total_optimal += cur_welfare - penalty_coeff_array[cur_index] * cur_overexceed_time
                schedule_output += "X[%d] = 1\n" % cur_index
                schedule_output += "Tau[%d] = %d\n" % (cur_index, cur_overexceed_time)
                for l,t in cur_scheduling:
                    cloudlet_counter[l] += 1
                    remain_workload_table[l, t] -= cur_workload
                    power_threshold -= power_range_array[l] * cur_workload
                    schedule_output += "(cloudlet %d at time %d)" % (l, t)
                    electricity_counter[t] += power_range_array[l] * cur_workload
                schedule_output += "\n\n"
            else:
                # 拒绝该任务
                pass

        # 输出最后结果
        if power_threshold < 0:
            total_inner_battery = abs(0 - power_threshold)
        else:
            total_inner_battery = 0
        end = time.clock()

        self.preempt_solution["optimal"] = total_optimal - inner_power_percost * total_inner_battery
        self.preempt_solution["accepted_rate"] = accepted_task_count / self.task_count
        self.preempt_solution["execute_time"] = end - start
        self.preempt_solution["Ug"] = total_inner_battery
        self.preempt_solution["cloudlet_counter"] = cloudlet_counter.most_common()
        self.preempt_solution["electricity_counter"] = electricity_counter.values()
        # self.preempt_solution["schedule"] = schedule_output

    # 对每个task到来时进行优化，得到局部最优
    def get_best_online(self):
        start = time.clock()
        # Beta[l] 常量，表示第h个cloudlet的power范围
        power_range_array = np.array([(l.peak_power - l.idle_power) * l.PUE for l in self.cloudlet_list])
        # f[j] 常量，惩罚函数的系数
        penalty_coeff_array = np.array([t.welfare/(self.end_EDR - t.deadline) for t in self.task_list])
        # p 常量，本地电池用电成本
        inner_power_percost = 0.32

        # D' = D - E - T*sum(N[l] * Pidle[l]) * PUE[l]
        power_threshold = self.original_energy - self.energy_cut - self.end_EDR * sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)
        # R[l, t] 表示已经第l个cloudlet在第t时刻剩余可用的workload
        remain_workload_array = np.array([l.servers_amount for l in self.cloudlet_list], dtype=np.int32)
        remain_workload_table = remain_workload_array.reshape(self.cloudlet_count, 1)
        remain_workload_table = np.repeat(remain_workload_table, self.end_EDR, axis=1)
        # Ug 表示总共需要的本地电量
        total_inner_battery = 0
        # 每个cloudlet调度总timeslot统计
        cloudlet_counter = Counter()
        # 记录每个任务j的具体调度实现
        schedule_output = str()
        # 总体最优值
        total_optimal = 0
        # 接受的任务总数，用以计算接受率
        accepted_task_count = 0
        # 统计每个时间片内的耗电量
        electricity_counter = Counter()
        # 每个时间片t内都至少有所有cloudlet空耗的电量
        for t in range(self.end_EDR):
            electricity_counter[t] += sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)

        # 循环j次，每次只对第j个任务求最优解
        for cur_task in self.task_list:
            model = Model("online local optimization")

            schedule_task = {}

            # 设置model的决策变量
            # task_choose = Xj, 表示第j个task是否选择被执行
            task_choose = model.addVar(vtype='B', name='X[%d]' % cur_task.index)
            # overexceed_time = Tau(j)，表示第j个task的超时时长
            overexceed_time = model.addVar(vtype='I', lb=0.0, ub=self.end_EDR-cur_task.deadline-1,
                                              name='Tau[%d]' % cur_task.index)
            # inner_battery = Utg, 表示第j个任务执行需要的本地电量
            inner_battery = model.addVar(vtype='C', lb=0.0, name='Ug[%d]' % cur_task.index)
            for l in range(self.cloudlet_count):
                for t in range(cur_task.arrive, self.end_EDR):
                    # schedule_task = Yjl(t), 表示第j个task在第t时间片中是否在第l个cloudlet中执行
                    schedule_task[l, t] = model.addVar(vtype='B', name='Y[%d, %d, %d]' % (cur_task.index, l, t))

            # 以下为约束条件
            for t in range(cur_task.arrive, self.end_EDR):
                for l in range(self.cloudlet_count):
                    model.addCons(cons=cur_task.workload * schedule_task[l, t] <= remain_workload_table[l, t],
                                  name='3a constrain')
                model.addCons(cons=quicksum(schedule_task[l, t]
                                            for l in range(self.cloudlet_count))
                                   <= 1,
                              name='3d constrain')
            for t in range(cur_task.arrive, self.end_EDR):
                model.addCons(cons=(t * quicksum(schedule_task[l, t]
                                                 for l in range(self.cloudlet_count))
                                    <= cur_task.deadline + overexceed_time),
                              name='3c constrain')
            model.addCons(cons=cur_task.time_need * task_choose
                               == quicksum(schedule_task[l, t]
                                           for l in range(self.cloudlet_count)
                                           for t in range(cur_task.arrive, self.end_EDR)),
                          name='3e constrain')
            model.addCons(cons=(quicksum(power_range_array[l] * schedule_task[l, t] * cur_task.workload
                                         for l in range(self.cloudlet_count)
                                         for t in range(cur_task.arrive, self.end_EDR))
                                <= power_threshold + total_inner_battery + inner_battery),
                          name='3b constrain')
            model.setObjective(coeffs=cur_task.welfare * task_choose - penalty_coeff_array[cur_task.index] * overexceed_time
                               - inner_power_percost * inner_battery,
                               sense='maximize')

            model.optimize()

            # 记录局部最优解相关信息并更新全局变量
            # if model.getStatus() == "optimal":
            total_optimal += model.getObjVal()
            X_j = int(round(model.getVal(task_choose)))
            Tau_j = int(round(model.getVal(overexceed_time)))
            Ug_j = model.getVal(inner_battery)
            total_inner_battery += Ug_j

            if X_j == 1:
                accepted_task_count += 1
                schedule_output += "X[%d] = 1\n" % cur_task.index
                schedule_output += "Tau[%d] = %d\n" % (cur_task.index, Tau_j)
                for t in range(cur_task.arrive, cur_task.deadline + Tau_j + 1):
                    for l in range(self.cloudlet_count):
                        Y_jlt = model.getVal(schedule_task[l, t])
                        if int(round(Y_jlt)) == 1:
                            schedule_output += "(cloudlet %d at time %d)" % (l, t)
                            cloudlet_counter[l] += 1
                            # 更新Rl[t], D',约束接下来的task
                            remain_workload_table[l, t] -= cur_task.workload
                            power_threshold -= power_range_array[l] * cur_task.workload
                            electricity_counter[t] += power_range_array[l] * cur_task.workload
                schedule_output += "\n\n"

        # 结束j轮局部循环优化，输出最后结果
        end = time.clock()

        self.online_solution["optimal"] = total_optimal
        self.online_solution["accepted_rate"] = accepted_task_count / self.task_count
        self.online_solution["execute_time"] = end - start
        self.online_solution["Ug"] = total_inner_battery
        self.online_solution["cloudlet_counter"] = cloudlet_counter.most_common()
        self.online_solution["electricity_counter"] = electricity_counter.values()
        # self.online_solution["schedule"] = schedule_output

    # 已知所有task情况，进行全局优化
    def get_best_offline(self):
        start = time.clock()
        model = Model("offline global optimization")
        # 如果一个解找到后遍历10000节点仍没有新的，就停止
        model.setLongintParam('limits/stallnodes', 10000)

        # task_choose = Xj, 表示第j个task是否选择被执行
        task_choose = {}
        # schedule_task = Yjl(t), 表示第j个task在第t时间片中是否在第l个cloudlet中执行
        schedule_task = {}
        # inner_battery = Utg, 表示第t个时间片需要的内部电池用电量
        inner_battery = 0
        # overexceed_time = Tau(j)，表示第j个task的超时时长
        overexceed_time= {}

        # w[j]
        time_need_array = np.zeros(self.task_count, dtype=np.int32)
        # lambda[j]
        workload_array = np.zeros(self.task_count, dtype=np.int32)
        # d[j]
        deadline_array = np.zeros(self.task_count, dtype=np.int32)
        # b[j]
        welfare_array = np.zeros(self.task_count, dtype=np.float64)
        # f[j]
        penalty_coeff_array = np.zeros(self.task_count, dtype=np.float64)
        # a[j]
        arrive_array = np.zeros(self.task_count, dtype=np.int32)

        for index, t in enumerate(self.task_list):
            time_need_array[index] = t.time_need
            workload_array[index] = t.workload
            deadline_array[index] = t.deadline
            welfare_array[index] = t.welfare
            penalty_coeff_array[index] = t.welfare / (self.end_EDR - t.deadline)
            arrive_array[index] = t.arrive

        servers_amount_array = np.zeros(self.cloudlet_count, dtype=np.int32)
        idle_power_array = np.zeros(self.cloudlet_count, dtype=np.float64)
        peak_power_array = np.zeros(self.cloudlet_count, dtype=np.float64)
        PUE_array = np.zeros(self.cloudlet_count, dtype=np.float64)
        for index, c in enumerate(self.cloudlet_list):
            servers_amount_array[index] = c.servers_amount
            idle_power_array[index] = c.idle_power
            peak_power_array[index] = c.peak_power
            PUE_array[index] = c.PUE

        # D' = D - E - T*sum(N[l] * Pidle[l]) * PUE[l]
        power_threshold = self.original_energy - self.energy_cut - self.end_EDR * sum(
            l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)
        # beta[l] = (Peak_power[l] - Idle_power[l]) * PUE[l]
        power_range_array = (peak_power_array - idle_power_array) * PUE_array
        # p 表示本地用电的单位价格
        inner_power_percost = 0.32


        # 这里的一些命名规范完全参考论文中的model
        for j in range(self.task_count):
            task_choose[j] = model.addVar(vtype='B', name='X[%d]' % j)
            overexceed_time[j] = model.addVar(vtype='I', lb=0.0, ub=self.end_EDR-deadline_array[j]-1,
                                              name='Tau[%d]' % j)
            for l in range(self.cloudlet_count):
                for t in range(arrive_array[j], self.end_EDR):
                    schedule_task[j, l, t] = model.addVar(vtype='B', name='Y[%d, %d, %d]' % (j, l, t))
        inner_battery = model.addVar(vtype='C', lb=0.0, name='Ug')

        # 以下为model约束条件
        for t in range(self.end_EDR):
            for l in range(self.cloudlet_count):
                model.addCons(cons=(quicksum(workload_array[j] * schedule_task[j, l, t]
                                             for j in range(self.task_count) if (j, l, t) in schedule_task)
                                    <= servers_amount_array[l]),
                              name='3a constrain')
        for j in range(self.task_count):
            for t in range(arrive_array[j], self.end_EDR):
                model.addCons(cons=(quicksum(schedule_task[j, l, t]
                                             for l in range(self.cloudlet_count))
                                    <= 1),
                              name='3d constrain')
        for j in range(self.task_count):
            for t in range(arrive_array[j], self.end_EDR):
                model.addCons(cons=(t * quicksum(schedule_task[j, l, t]
                                                 for l in range(self.cloudlet_count))
                                    <= deadline_array[j] + overexceed_time[j]),
                          name='3c constrain')

        for j in range(self.task_count):
            model.addCons(cons=(time_need_array[j] * task_choose[j]
                                == quicksum(schedule_task[j, l, t]
                                            for l in range(self.cloudlet_count)
                                            for t in range(arrive_array[j], self.end_EDR))),
                          name='3e constrain')
        model.addCons(cons=(quicksum(power_range_array[l] * workload_array[j] * schedule_task[j, l, t]
                                     for j in range(self.task_count)
                                     for l in range(self.cloudlet_count)
                                     for t in range(arrive_array[j], self.end_EDR))
                            <= power_threshold + inner_battery),
                      name='3b constrain')

        # 这里没有考虑pi, p, theta_i的值！！！
        model.setObjective(coeffs=(quicksum((welfare_array[j] * task_choose[j]) - (penalty_coeff_array[j] * overexceed_time[j])
                                            for j in range(self.task_count))
                                   - inner_power_percost * inner_battery),
                           sense='maximize')

        model.optimize()
        end = time.clock()
        # 显示最优解的相关信息
        cloudlet_counter = Counter()
        # 统计每个时间片内的耗电量
        electricity_counter = Counter()
        # 每个时间片t内都至少有所有cloudlet空耗的电量
        for t in range(self.end_EDR):
            electricity_counter[t] += sum(l.servers_amount * l.idle_power * l.PUE for l in self.cloudlet_list)

        accepted_task_count = 0
        schedule_output = str()
        for j in range(self.task_count):
            X_j = int(round(model.getVal(task_choose[j])))
            Tau_j = int(round(model.getVal(overexceed_time[j])))

            if X_j == 1:
                accepted_task_count += 1
                schedule_output += "X[%d] = 1\n" % j
                schedule_output += "Tau[%d] = %d\n" % (j, Tau_j)
                cur_arrive = self.task_list[j].arrive
                cur_deadline = self.task_list[j].deadline
                for t in range(cur_arrive, cur_deadline + Tau_j + 1):
                    for l in range(self.cloudlet_count):
                        Y_jlt = model.getVal(schedule_task[j, l, t])
                        # print(Y_jlt)
                        if int(round(Y_jlt)) == 1:
                            schedule_output += "(cloudlet %d at time %d)" % (l, t)
                            cloudlet_counter[l] += 1
                            electricity_counter[t] += power_range_array[l] * self.task_list[j].workload
                schedule_output += "\n\n"
        self.offline_solution["optimal"] = model.getObjVal()
        self.offline_solution["accepted_rate"] = accepted_task_count / self.task_count
        self.offline_solution["execute_time"] = end - start
        self.offline_solution["Ug"] = model.getVal(inner_battery)
        self.offline_solution["cloudlet_counter"] = cloudlet_counter.most_common()
        self.offline_solution["electricity_counter"] = electricity_counter.values()
        self.offline_solution["primal-dual gap"] = model.getGap()
        # self.offline_solution["schedule"] = schedule_output

    # 随机生成cloudlet
    def generate_cloudlet_list(self, servers_amount_mid=20, idle_power=0.01, peak_power=0.03):
        # 假设已经知道U[l]/L[l]的具体数值
        # time_need_array = np.array([j.time_need for j in self.task_list])
        # max_time_need = time_need_array.max()
        # min_time_need = time_need_array.min()

        result_list = []
        # 电量一律采用kwh作为单位1，time slot=10min
        # P[idle] = 60w, P[peak] = 180w 转化为1 time slot内的电量即为0.01kwh, 0.03kwh
        self.cloudlet_info["server_mid"] = servers_amount_mid
        self.cloudlet_info["server_range"] = (0.8, 1.2)
        self.cloudlet_info["idle_power"] = idle_power
        self.cloudlet_info["peak_power"] = peak_power
        self.cloudlet_info["PUE_range"] = (1.9, 2.5)
        # self.cloudlet_info["upper"] = price * max_time_need
        # self.cloudlet_info["lower"] = price * min_time_need
        # self.cloudlet_info["UlLl_ratio"] = max_time_need / min_time_need

        for i in range(self.cloudlet_count):
            cur_servers_amount = random.randint(int(0.8 * servers_amount_mid), int(1.2 * servers_amount_mid))

            cur_PUE = random.uniform(1.9, 2.5)

            cur_cloudlet = Cloudlet(index=i, servers_amount=cur_servers_amount,
                                    idle_power=idle_power, peak_power=peak_power, PUE=cur_PUE)
            result_list.append(cur_cloudlet)
        return result_list

    def print_cloudlet_list(self):
        output = str()
        for cloudlet in self.cloudlet_list:
            output += 'cloudlet[%d]:servers_amount=%d, idle_power=%f, peak_power=%f, PUE=%f\n' % \
                      (cloudlet.index, cloudlet.servers_amount, cloudlet.idle_power, cloudlet.peak_power, cloudlet.PUE)
        print(output)
        self.cloudlet_info["data"] = output

    # 随机生成task
    def generate_task_list(self, workload_max=20):
        result_list = []
        self.task_info["workload_max"] = workload_max
        self.task_info["workload_range"] = (0.4, 1)
        self.task_info["elastic_time"] = 0
        # 90 workload * 6 h = $112, 0.034表示1 workload*1 timeslot的价值
        self.task_info["welfare_ratio"] = (0.01, 0.04)

        # 用泊松分布表示task的arrive_time
        poisson_array = np.random.poisson(lam=((self.end_EDR-1)/self.task_count), size=self.task_count)
        while poisson_array.sum() > self.end_EDR - 1:
            poisson_array = np.random.poisson(lam=((self.end_EDR-1)/self.task_count), size=self.task_count)

        for i in range(self.task_count):
            cur_price = random.uniform(0.01, 0.04)
            cur_workload = random.randint(int(0.4 * workload_max), workload_max)

            cur_arrive = poisson_array[:i+1].sum()
            cur_time_need = random.randint(1, self.end_EDR - cur_arrive)
            # 这里默认从arrive_time处开始执行 而不是从arrive_time+1开始执行
            normal_finish_time = cur_arrive + cur_time_need - 1
            # increase_time = random.randint(0, self.end_EDR - normal_finish_time)
            cur_deadline = normal_finish_time + random.randint(0, (self.end_EDR - 1 - normal_finish_time))
            # cur_deadline = normal_finish_time

            cur_welfare = cur_price * cur_workload * cur_time_need

            cur_task = Task(index=i, workload=cur_workload, arrive=cur_arrive, time_need=cur_time_need,
                            deadline=cur_deadline, welfare=cur_welfare, unit_price=cur_price)
            result_list.append(cur_task)
        return result_list

    def print_task_list(self):
        output = str()
        for task in self.task_list:
            output += 'task[%d]:workload=%d, arrive=%d, time_need=%d, deadline=%d, welfare=%f, unit_price=%f\n' % \
                      (task.index, task.workload, task.arrive, task.time_need, task.deadline, task.welfare, task.unit_price)
        print(output)
        self.task_info["data"] = output

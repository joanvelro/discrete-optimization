#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
Item = namedtuple("Item", ['index', 'value', 'weight', 'density'])
import numpy as np
from operator import attrgetter
np.random.seed = 0

def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    firstLine = lines[0].split()
    item_count = int(firstLine[0])
    capacity = int(firstLine[1])

    print('\n')
    print('Input data')
    print('--------')
    print('Capacity: ', capacity)
    print('Elements: ', item_count)

    items = []

    for i in range(1, item_count + 1):
        line = lines[i]
        parts = line.split()
        value = int(parts[0])
        weight = int(parts[1])
        density = value / weight
        items.append(Item(i - 1, value, weight, density))

    for item in items:
        print('Item {}, Value:{}, Weight:{}'.format(item.index, item.value, item.weight))
    print('\n')
    naive_estimate = sum([item.value for item in items])

    # ---------- SOLVERS  -------------------
    def greedy_algorithm(items):
        # a trivial algorithm for filling the knapsack
        # it takes items in-order until the knapsack is full
        value = 0
        weight = 0
        taken = [0]*len(items)

        for item in items:
            if weight + item.weight <= capacity:
                taken[item.index] = 1
                value += item.value
                weight += item.weight
        return value, taken

    def random_search(items):
        print('\n')
        print('Random Search')
        print('-' * 30)
        optimal_solution = []
        feasible_solutions_set = []
        objective_values = []
        weight_values = []
        best_solution = 0
        set_already_searched = []
        # candidate solution
        n = len(items)
        flag_iter = True
        iter = 0
        discarded_solution_in_row = 0
        while flag_iter:
            iter += 1
            # generate a solution from random
            x = np.random.randint(low=0, high=2, size=n).tolist()
            if x not in set_already_searched:
                set_already_searched.append(x)
                print('Solutions explored:', len(set_already_searched))
                # initialize aux variables for value and weight of the solution
                total_value = 0
                total_weight = 0
                for i, a in enumerate(x):
                    if x[i] == 1:
                        total_value += items[i].value
                        total_weight += items[i].weight
                if 0 < total_weight <= capacity:
                    print('candidate solution:', x)
                    # feasible solution
                    print('\tIts a Feasible solution!')
                    print('\tTotal Value:', total_value)
                    print('\tTotal weight: {} < {}'.format(total_weight, capacity))
                    feasible_solutions_set.append(x)
                    objective_values.append(total_value)
                    weight_values.append(total_weight)
                    # check if improves the existing solutions
                    if total_value > best_solution:
                        print('\tIt it the new best solution :)')
                        best_solution = total_value
                        optimal_solution = x
                    else:
                        print('added solution for discard')
                        discarded_solution_in_row += 1
                        print('\tDiscard solution :(')

                if len(set_already_searched) == 2 ** n:
                    flag_iter = False
                if discarded_solution_in_row == 100:
                    print('Reached limit of discarded solutions')
                    flag_iter = False

        if optimal_solution is not None:
            print('Optimal solution:', optimal_solution)
            print('Optimal value:', best_solution)
            print('No. Iterations: ', iter)
        else:
            print('No optimal solution founded')
        if not feasible_solutions_set:
            print('No feasible solution founded')
        print('--------------------\n')
        return best_solution, optimal_solution

    items_by_value = sorted(items, key=attrgetter('value'), reverse=True)
    objective_value_2, taken_2 = greedy_algorithm(items=items_by_value)

    items_by_density = sorted(items, key=attrgetter('density'), reverse=True)
    objective_value_3, taken_3 = greedy_algorithm(items=items_by_density)

    items_by_weight = sorted(items, key=attrgetter('weight'), reverse=False)
    objective_value_4, taken_4 = greedy_algorithm(items=items_by_weight)

    def take_second(elem):
        return elem[1]

    GS_objective_values = [('Greedy sorted by value  ', objective_value_2, taken_2),
                           ('Greedy sorted by density', objective_value_3, taken_3),
                           ('Greedy sorted by weight ', objective_value_4, taken_4)]

    GS_sorted_list = sorted(GS_objective_values, key=take_second, reverse=True)

    for result in GS_sorted_list:
        print('{} value: {} solution: {}'.format(result[0], result[1], result[2]))

    # optimal_solution_value, optimal_decision = random_search(items=items)

    # ---------- OUTPUT   -------------------
    def output_format(objective: int, taken: list):
        # prepare the solution in the specified output format
        output_data = str(objective) + ' ' + str(0) + '\n'
        output_data += ' '.join(map(str, taken))
        return output_data

    output_data = output_format(GS_sorted_list[0][1], GS_sorted_list[0][2])

    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')


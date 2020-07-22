from gurobipy import *
import itertools
from math import sqrt

def min_max_length_under_complete_graph(city_num, deliver_num, weight_metrix, TL):
    model = Model("TSP")
    model.setParam(GRB.Param.TimeLimit, TL)
    # Create variables
    x = {}
    for i in range(city_num):
        for j in range(city_num):
            for k in range(deliver_num):
                x[i, j, k] = model.addVar(vtype=GRB.BINARY, name='e_' + str(i) + '_' + str(j) + '_' + str(k))
    Q = model.addVar(name='Q')
    
    model.setObjective(Q, GRB.MINIMIZE)
    
    for k in range(deliver_num):
        model.addConstr(quicksum(x[0, j, k] for j in range(1, city_num)) == 1)
        model.addConstr(quicksum(x[i, 0, k] for i in range(1, city_num)) == 1)
    for i in range(1, city_num):
        model.addConstr(quicksum(x[i, j, k]
                                 for j in range(city_num)
                                 for k in range(deliver_num)) == 1
                        )
    for j in range(1, city_num):
        model.addConstr(quicksum(x[i, j, k]
                                 for i in range(city_num)
                                 for k in range(deliver_num)) == 1
                        )
    for r in range(1, city_num):
        for k in range(deliver_num):
            model.addConstr((quicksum(x[i, r, k] for i in range(city_num))
                             - quicksum(x[r, j, k] for j in range(city_num))) == 0
                            )
    model.addConstrs((x[i, i, k] == 0
                      for k in range(deliver_num)
                      for i in range(1, city_num)), name='C'
                     )
    for k in range(deliver_num):
        model.addConstr(quicksum(weight_metrix[i][j] * x[i, j, k]
                                 for i in range(city_num)
                                 for j in range(city_num)) <= Q)

    # Callback - use lazy constraints to eliminate sub-tours
    def subtourelim(model, where):
        if where == GRB.callback.MIPSOL:
            # make a list of edges selected in the solution
            for k in range(deliver_num):
                selected = []
                visited = set()
                for i in range(city_num):
                    sol = model.cbGetSolution([x[i, j, k] for j in range(city_num)])
                    new_selected = [(i, j) for j in range(city_num) if sol[j] > 0.5]
                    selected += new_selected

                    if new_selected:
                        visited.add(i)

                tour = subtour(selected, visited)

                if len(tour) < len(visited):
                    # add a subtour elimination constraint
                    expr = quicksum(x[i, j, k] for i, j in itertools.permutations(tour, 2))
                    model.cbLazy(expr <= len(tour) - 1)

    # Optimize model
    model.update()
    model.params.LazyConstraints = 1
    # model.optimize()
    model.optimize(subtourelim)

    node_mat = [[[0 for i in range(city_num)] for i in range(city_num)] for i in range(deliver_num)]
    for k in range(deliver_num):
        for i in range(city_num):
            for j in range(city_num):
                node_mat[k][i][j] = x[i, j, k].x

    allpath = []
    for k in range(deliver_num):
        path = []
        cnt = 0
        while True:
            path.append(cnt)
            for j in range(city_num):
                if node_mat[k][cnt][j] > 0.5:
                    cnt = j
                    break
            if cnt in path:
                path.append(cnt)
                break
        allpath.append(path)
    return allpath

def subtour(edges, visited):
    unvisited = list(visited)
    cycle = range(len(visited) + 1)
    selected = {}
    for x, y in edges:
        selected[x] = []
    for x, y in edges:
        selected[x].append(y)
    # print (selected)
    while unvisited:
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for j in selected[current] if j in unvisited]
        if len(cycle) > len(thiscycle):
            cycle = thiscycle
    return cycle
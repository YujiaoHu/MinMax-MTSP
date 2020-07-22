"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch
import multiprocessing
import time
import numpy as np

C = 100000
const_process_num = 26


def Euclidean_distance(coords):
    city_square = torch.sum(coords ** 2, dim=1, keepdim=True)
    city_square_tran = torch.transpose(city_square, 1, 0)
    cross = -2 * torch.matmul(coords, torch.transpose(coords, 1, 0))
    dist = city_square + city_square_tran + cross
    dist = torch.sqrt(dist)
    for m in range(dist.size(0)):
        dist[m, m] = 0.0
    dist = dist * C
    return dist.long().numpy()


def create_data_model(coords, anum):
    """Stores the data for the problem."""
    data = {}
    data['num_vehicles'] = anum
    data['depot'] = 0
    data['distance_matrix'] = []
    dist = Euclidean_distance(coords)
    cnum = coords.size(0)
    for c in range(cnum):
        data['distance_matrix'].append(list(dist[c]))
    return data


def print_solution(data, manager, routing, solution):
    anum = data['num_vehicles']
    tourlen = torch.zeros(anum)
    """Prints solution on console."""
    # max_route_distance = 0
    tours = []
    for vehicle_id in range(data['num_vehicles']):
        atour = []
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            atour.append(manager.IndexToNode(index))
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        atour.append(manager.IndexToNode(index))
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance/C)
        tourlen[vehicle_id] = route_distance/C
        tours.append(atour)
        # print(plan_output)
        # max_route_distance = max(route_distance, max_route_distance)
    # print('Maximum of the route distances: {}m'.format(max_route_distance/100000))
    return tourlen, tours

def solve_instance_vrp(inputs):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    coords = inputs[0]
    anum = inputs[1]
    if len(inputs) > 2:
        timelimit = int(inputs[2]) + 1
        switch_timeLimit = True
    else:
        timelimit = 60
        switch_timeLimit = False

    start_time = time.time()
    data = create_data_model(coords, anum)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        10000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Setting time limitation
    search_parameters.time_limit.seconds = timelimit
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    tourlen, tours = print_solution(data, manager, routing, solution)
    return [tourlen, tours]


def ortool_baseline_singleTrack(coords, anum):
    # print(coords.size())
    device = coords.device
    coords = coords.cpu()
    batch_size = coords.size(0)
    tourlen = []
    replan_tours = []
    for b in range(batch_size):
        result = solve_instance_vrp([coords[b], anum])
        tourlen.append(result[0])
        replan_tours.append(result[1])
    tourlen = torch.stack(tourlen, dim=0)
    return tourlen.to(device), replan_tours


def orplanning_under_timeLimitation_singTrack(coords, anum, tusage):
    device = coords.device
    coords = coords.cpu()
    batch_size = coords.size(0)
    tourlen = []
    replan_tours = []
    for b in range(batch_size):
        result = solve_instance_vrp([coords[b], anum, tusage])
        tourlen.append(result[0])
        replan_tours.append(result[1])
    tourlen = torch.stack(tourlen, dim=0)
    return tourlen.to(device), replan_tours


# def orplanning_under_inital_solution(coords, anum, tusage, initial_solution):
#     device = coords.device
#     coords = coords.cpu()
#     batch_size = coords.size(0)
#     pool = multiprocessing.Pool(processes=26)
#     multi_inputs = []
#     for b in range(batch_size):
#         mptour = []
#         for a in range(anum):
#             # print(initial_solution[b][0][a][1:-1])
#             atour = torch.tensor(initial_solution[b][0][a][1:-1]).long().cpu()
#             atour = list(np.array(atour))
#             mptour.append(atour)
#         multi_inputs.append([coords[b], anum, tusage, mptour])
#     result = pool.map(solve_vrp_with_inital_solution, multi_inputs)
#     pool.close()
#     tourlen = []
#     duration = []
#     for b in range(batch_size):
#         tourlen.append(result[b][0])
#         duration.append(result[b][1])
#     tourlen = torch.stack(tourlen, dim=0)
#     duration = torch.tensor(duration)
#     return tourlen.to(device), duration.to(device)


def orplanning_under_inital_solution_singleTrack(coords, anum, tusage, initial_solution):
    device = coords.device
    coords = coords.cpu()
    batch_size = coords.size(0)
    tourlen = []
    replan_tours = []
    for b in range(batch_size):
        mptour = []
        for a in range(anum):
            # print(initial_solution[b][0][a][1:-1])
            atour = torch.tensor(initial_solution[b][0][a][1:-1]).long().cpu()
            atour = list(np.array(atour))
            mptour.append(atour)
        result = solve_vrp_with_inital_solution([coords[b], anum, tusage, mptour])
        tourlen.append(result[0])
        replan_tours.append(result[1])
    tourlen = torch.stack(tourlen, dim=0)
    return tourlen.to(device), replan_tours


def solve_vrp_with_inital_solution(inputs):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    coords = inputs[0]
    anum = inputs[1]
    timelimit = int(inputs[2])
    init_solution = inputs[3]

    # print(init_solution)
    start_time = time.time()
    data = create_data_model(coords, anum)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Distance constraint.
    dimension_name = 'Distance'
    routing.AddDimension(
        transit_callback_index,
        0,  # no slack
        10000000,  # vehicle maximum travel distance
        True,  # start cumul to zero
        dimension_name)
    distance_dimension = routing.GetDimensionOrDie(dimension_name)
    distance_dimension.SetGlobalSpanCostCoefficient(100)

    # Setting initial solutions
    initial_solution = routing.ReadAssignmentFromRoutes(init_solution, True)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Setting time limitation
    search_parameters.time_limit.seconds = timelimit

    # Solve the problem.
    solution = routing.SolveFromAssignmentWithParameters(initial_solution, search_parameters)

    # Print solution on console.
    if solution:
        tourlen, tours = print_solution(data, manager, routing, solution)
        tusage = time.time() - start_time
        # print(tourlen)
        # print("--------------------")
        return [tourlen, tours]

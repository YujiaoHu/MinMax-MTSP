"""Simple travelling salesman problem between cities."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import torch


C = 100000

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


def create_data_model(coords):
    """Stores the data for the problem."""
    data = {}
    data['distance_matrix'] = []
    dist = Euclidean_distance(coords)
    cnum = coords.size(0)
    for c in range(cnum):
        data['distance_matrix'].append(list(dist[c]))
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data


def print_solution(manager, routing, assignment):
    """Prints assignment on console."""
    # print('Objective: {} miles'.format(assignment.ObjectiveValue()/C))
    index = routing.Start(0)
    # plan_output = 'Route for vehicle 0:\n'
    # route_distance = 0
    vehicleTour = []
    while not routing.IsEnd(index):
        vehicleTour.append(manager.IndexToNode(index))
        # plan_output += ' {} ->'.format(manager.IndexToNode(index))
        # previous_index = index
        index = assignment.Value(routing.NextVar(index))
        # route_distance += routing.GetArcCostForVehicle(previous_index, index, 0)
    # plan_output += ' {}\n'.format(manager.IndexToNode(index))
    vehicleTour.append(manager.IndexToNode(index))
    # # print(plan_output)
    # plan_output += 'Route distance: {}miles\n'.format(route_distance*1.0/C)
    return assignment.ObjectiveValue()/C, vehicleTour


def entrance(coords):
    # print("setting finish")

    """Entry point of the program."""
    # Instantiate the data problem.
    data = create_data_model(coords)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                           data['num_vehicles'], data['depot'])

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = 10

    # Solve the problem.
    assignment = routing.SolveWithParameters(search_parameters)
    # print("computing finish")

    # Print solution on console.
    if assignment:
        return print_solution(manager, routing, assignment)
    else:
        print("!!!ortool error: ", coords)


if __name__ == '__main__':
    coord = torch.rand(20, 2)
    entrance(coord)

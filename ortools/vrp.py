"""Vehicles Routing Problem (VRP)."""

from __future__ import print_function
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import torch

C = 10000
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


def create_data_model(cnum, anum):
    """Stores the data for the problem."""
    data = {}
    data['num_vehicles'] = anum
    data['depot'] = 0
    data['distance_matrix'] = []
    coords = torch.rand(cnum, 2)
    data['coords'] = coords
    dist = Euclidean_distance(coords)
    cnum = coords.size(0)
    for c in range(cnum):
        data['distance_matrix'].append(list(dist[c]))
    return data, coords


def computing_tourlen(data, tour):
    atour = torch.tensor(tour)
    steps = atour.size(0)
    x = torch.zeros(steps + 1).long()
    y = x.clone()
    x[1:] = atour
    y[:steps] = atour

    coords = data['coords']
    xcoord = torch.gather(coords[:, :2], 0, x.unsqueeze(1).repeat(1, 2))
    ycoord = torch.gather(coords[:, :2], 0, y.unsqueeze(1).repeat(1, 2))
    temp = xcoord - ycoord
    return torch.sum(torch.sqrt(temp[:, 0] ** 2 + temp[:, 1] ** 2))

def print_solution(data, manager, routing, solution):
    """Prints solution on console."""
    tourlen = torch.zeros(data['num_vehicles'])
    max_route_distance = 0
    for vehicle_id in range(data['num_vehicles']):
        tour = []
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        while not routing.IsEnd(index):
            tour.append(manager.IndexToNode(index))
            plan_output += ' {} -> '.format(manager.IndexToNode(index))
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += '{}\n'.format(manager.IndexToNode(index))
        plan_output += 'Distance of the route: {}m\n'.format(route_distance/C)
        # print(plan_output)
        tourlen[vehicle_id] = computing_tourlen(data, tour)
        # print(tour, tourlen[vehicle_id])
        # max_route_distance = max(route_distance, max_route_distance)
    # print('Maximum of the route distances: {}m'.format(max_route_distance/C))
    return tourlen


def entrance(cnum, anum, timeLimitation=1800):
    """Solve the CVRP problem."""
    # Instantiate the data problem.
    data, coords = create_data_model(cnum, anum)

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
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.time_limit.seconds = timeLimitation
    
    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    # Print solution on console.
    if solution:
        tourlen = print_solution(data, manager, routing, solution)
        return tourlen, coords


if __name__ == '__main__':
    for i in range(200):
        main()
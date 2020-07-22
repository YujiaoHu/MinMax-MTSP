import random
from math import sqrt

class getdata():
    def __init__(self, city_num, dimension):
        self.city_num = city_num
        self.dimension = dimension
        self.cities = self.generate_coordinate(self.city_num, self.dimension)
        self.weights_metrix = self.distance()

    def generate_coordinate(self, num, dimension=2):
        num = int(num)
        if dimension == 2:
            cities = [[random.uniform(-1, 1),
                       random.uniform(-1, 1)] for i in range(num)]
        if dimension == 3:
            cities = [[random.uniform(-1, 1),
                       random.uniform(-1,1),
                       random.uniform(-1,1)] for i in range(num)]
        return cities

    def distance(self):
        nodes_mat = [[0 for i in range(0, self.city_num)] for i in range(0, self.city_num)]
        if self.dimension == 2:
            for i in range(0, self.city_num):
                for j in range(i, self.city_num):
                    d = sqrt(pow((self.cities[i][0] - self.cities[j][0]), 2)
                             + pow((self.cities[i][1] - self.cities[j][1]), 2))
                    nodes_mat[i][j], nodes_mat[j][i] = d, d
        if self.dimension == 3:
            for i in range(0, self.city_num):
                for j in range(i, self.city_num):
                    d = sqrt(pow((self.cities[i][0] - self.cities[j][0]), 2)
                             + pow((self.cities[i][1] - self.cities[j][1]), 2)
                             + pow((self.cities[i][2] - self.cities[j][2]), 2)
                             )
                    nodes_mat[i][j], nodes_mat[j][i] = d, d

        for c in range(self.city_num):
            nodes_mat[c][c] = 10000
        return nodes_mat

if __name__ == '__main__':
    dataset = getdata(10, 2)
    weight = dataset.weights_metrix

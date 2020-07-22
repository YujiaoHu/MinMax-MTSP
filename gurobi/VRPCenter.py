from models.min_max import min_max_length_under_complete_graph as min_max
from data.generate_Enclic_data_complete_graph import getdata as getdataEC

class VRPCenter:
    def __init__(self, city_num=0, deliver_num=0, dimension=2):
        self.dimension = dimension
        self.delivers = deliver_num
        self.city_num = city_num

        data_set = getdataEC(city_num=self.city_num, dimension=self.dimension)
        self.cities = data_set.cities
        self.weight_metrix = data_set.weights_metrix

    def start(self, TL):
        allpath = min_max(city_num=self.city_num,
                          deliver_num=self.delivers,
                          weight_metrix=self.weight_metrix,
                          TL=TL)
        return allpath


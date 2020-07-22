# MinMax-MTSP
Implementation of ["A Reinforcement Learning Approach for Optimizing Multiple Traveling Salesman Problems over Graphs"](https://www.sciencedirect.com/science/article/abs/pii/S0950705120304445?dgcid=rss_sd_all), which is accepted at "Knowledge-based System". If this code is useful for your work, please cite our paper:

		@article{hu2020reinforcement,
		  title={A reinforcement learning approach for optimizing multiple traveling salesman problems over graphs},
		  author={Hu, Yujiao and Yao, Yuan and Lee, Wee Sun},
		  journal={Knowledge-Based Systems},
		  pages={106244},
		  year={2020},
		  publisher={Elsevier}
		}

## Dependencies
* Python >= 3.6
* Numpy
* Google OR-Tools
* Gurobi
* PyTorch
* tqdm
* TensorboardX

## Usage
### Generating Data
Codes under `gurobi` and `ortools` can genearte data computed by `Gurobi` and `Google OR-Tools` respectively.

You are expected to generate validate/test data before running the training codes.

### Training Model
The training codes are given under `partition`

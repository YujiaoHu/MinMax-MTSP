## Training
```python
python train_hop.py
```

You are also allowed to modify paprameters in the `options.py` or run
```python
python train_hop --anum 5 --cnum 100
```

## Note
The models are able to converge quickly after training for 2k iterations, but if you want to get better performance on large-scale instances, just train for a longer time (approximately 40k iterations)

## Reference
The structure of `Graph Neural Network` is motivated by [Deep Graphical Feature Learning for the Feature Matching Problem](https://github.com/zzhang1987/Deep-Graphical-Feature-Learning)

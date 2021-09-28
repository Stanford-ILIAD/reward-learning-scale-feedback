Companion code to CoRL 2021 paper:  
Nils Wilde, Erdem Bıyık, Dorsa Sadigh, Stephen L. Smith. **"Learning Reward Functions from Scale Feedback"**. *5th Conference on Robot Learning (CoRL)*, London, UK, Nov. 2021.

This code learns reward functions from scale feedback in various tasks and compares it to learning from soft choice feedback. The code also includes active query generation methods in addition to random querying: information gain maximization and minimax regret.

The codes for the interface of the user studies are excluded, but the simulation environments are still provided.

## Dependencies
You need to have the following libraries with [Python3](http://www.python.org/downloads):
- [matplotlib](http://matplotlib.org/)
- [NumPy](http://www.numpy.org/)
- [OpenAI Gym](http://gym.openai.com)
- [pyglet](http://bitbucket.org/pyglet/pyglet/wiki/Home)
- [SciPy](http://www.scipy.org/)
- [theano](http://deeplearning.net/software/theano/)
- [pandas](https://pandas.pydata.org/)

## Running
config.py includes the parameters for the experiments. After setting these parameters, you simply run:
```python
	python run.py
```
After the runs are completed, you can visualize the results by running:
```python
	python plot_data.py
```

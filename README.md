# ContraBAR: Contrastive Bayes-Adaptive Deep RL


> ! Important !
> 
> If you use this code with your own environments, 
> make sure to not use `np.random` in them 
> (e.g. to generate the tasks) because it is not thread safe 
> (and not using it may cause duplicates across threads).
> Instead, use the python native random function. 
> For an example see
> [here](https://github.com/lmzintgraf/varibad/blob/master/environments/mujoco/ant_goal.py#L38).

### Requirements

We use PyTorch for this code, and log results using TensorboardX.

Because the different mujoco environments require different versions, we supply
three seperate requirement files:
 - `requirements_131.txt`
 - `requirements_150.txt`
 - `requirements_200.txt`

For `mujoco131`, `mujoco150` and `mujoco200` respectively. 

##### We note that non-mujoco environments should all work with `requirements_200`.

#### Mujoco:
For the MuJoCo experiments, you need to install MuJoCo.
Make sure you have the right MuJoCo version:
- For the Cheetah and Ant environments, use `mujoco150`. 
(You can also use `mujoco200` except for AntGoal, 
because there's a bug which leads to 80% of the env state being zero).
- For Walker/Hopper, use `mujoco131`.

#### Reacher environment:

For the reacher environment, dm_control was slightly modified to include
transparent target, which replaces the target in the classic reacher env.
This code is in a fork of dm_control which is referenced in requirements_200.txt:
[Modified dm_control package](https://github.com/ec2604/dm_control)

#### Panda-gym environment:
 
For the custom_reach task for the panda-gym environment, panda-gym was
slightly modified - this modification is in a fork referenced in requirements_200.txt : 
[Modified panda-gym package](https://github.com/ec2604/panda-gym)
### Overview

The main training loop for ContraBAR can be found in `metalearner_cpc.py`,
the models are in `models/`, the CPC set-up and losses are in `cpc.py` and the RL algorithms in `algorithms/`.


### Running an experiment

To train contraBAR on the different environments use:
```
python main.py --env-type ENV
```
For example
```
python main.py --env-type pointrobot_contrabar
```


The results will by default be saved at `./logs`, 
but you can also pass a flag with an alternative directory using `--results_log_dir /path/to/dir`.

The default configs are in the `config/` folder. 
You can overwrite any default hyperparameters using command line arguments.

Results will be written to tensorboard event files, 
and some visualisations will be printed every now and then.

### Results

The results can be found in ```end_performance_per_episode``` and the script to generate them
is ```eval_test_perf```. 

For evaluating and visualizing results the ```eval_script``` and
```viz_script``` are available.

### Citation

### Communication
For any questions, please contact Era Choshen: ```erachoshen@gmail.com```
# CommonroadWrapper
This implementation provides some basic functions to wrap the Commonroad@TUM framework in an OpenAI Gym-like notation to be usable in a reinforcement learning setting.

For more information about Commonroad, please refer to:

https://commonroad.in.tum.de/

Required packages:
```
pip install commonroad-io commonroad-vehicle-models
```

## How it can be used

The scenarioGenerator function in /env allows to reproduce arbitrary highway environments which are preliminary constrained to straight lane settings. The configuration details can be chosen in env/config/.

To generate a basic highway environment:

1. Set the lane parameters in /env/config/scenarioConfig.yaml, the obstacle attribute is irrelevant here as the generator will only produce the scenario layout.

2. Additionally a planning problem can be defined with a goal region that should be reached in /env/config/planningProblem.yaml

3. Open a file and run

   ```
   from env.scenarioGenerator import create_env
   create_env("your-env-name")
   ```

The scenario can be found in /scenario/your-env-name.xml.

To run a interactive scenario:

1. Go back to the config file /env/config/scenarioConfig.yaml and add obstacles to the list of the "obstacle attribute" (copy the dictionary)
2. Specify the obstacles initial lane, x coordinate and speed (is given as percentage of the speed limit)
3. Check the main.py file for an example on how to run the scenario

## General Information

Current Limitations:

* only straight lanes can be generated
* car agents are controlled by the Intelligent Driver Model ([IDM](https://arxiv.org/abs/cond-mat/0002177))

Next implementation steps:

* Calibrating the IDM for aggressive, passive and normal driving behaviors
* Adding MOBIL Lane change model
* Adding reward, observation for .step() return as well as collision checks and terminal state checks 




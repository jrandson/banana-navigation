### Deep Reiforcement Learning

![An example of the environment to be solved](banana.gif)
#### The Environment

In this project, an agent is trained to navigate (and collect bananas!) in a large, square world.


A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. 
Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The pourpose of this project is train the agent so that it is capable to get an average score of +13 over 100 consecutive episodes.

### Basic setup

For this project, You can download the from the link below. You need only select the environment that matches your operating system:

* [Linux environment: click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* [Mac OSX environemnt: click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

Then, place the file in the p1_navigation/ folder in the DRLND GitHub repository, and unzip (or decompress) the file.

You are encouraged to use a virtual env and once you have setted it one, you can install the requirements to
this project by running `$ pip install -r requirements.txt`

After the instaltion is finished, run the main.py file by `$ python main.py` and observe the train model 
happening
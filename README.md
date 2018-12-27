# Reinforcement-Learning-Game

A reinforcement learning environment created to allow development of reinforcement learning algorithms, including basic solutions.

<img src="https://raw.githubusercontent.com/splovyt/Reinforcement-Learning-Game/master/docs/screenshot.png" height=300>

## The Challenge

Use my Bomberman-inspired game environment (Py3) to explore and train reinforcement learning algorithms (MCTS, DQN, Genetic Algorithms, and more) to develop an unbeatable AI agent.

Instructions on how to install this environment can be found below.

## State of my solutions

**A. Deep Q Network (DQN)**

Basic implementation is complete.


**B. Genetic Algorithms (GA) - In Progress**

Currently randomly plays games to collect data for future selection.


**C. Monte Carlo Tree Search (MCTS) - In Progress**

Highest potential for a good solutions (ref. AlphaGo and Chess), but still needs to be added. 


## Instructions on how to install the environment

1. Clone this repository
```git clone ...```

2. Install the ```requirements.txt``` or ```dev-requirements.txt``` file.

```pip install -r dev-requirements.txt```

or

```pipenv install -r dev-requirements.txt```

depending on which you are using.

3. Navigate to the ```MyBot.py``` file. This is where you should code your algorithm. You can run this file to show a battle of two bots with randomly selected actions. Spoiler alert: it's only a matter of time until each bot blows itself up..

- the standard option is to save the frames of the game and create a video at the end. The data will be saved in the ```data/{game_id}/``` folder.

- besides saving the visual representations of the game, a data driven representation can be generated using the ```game.get_status_dict()``` function at any time.

**Tip: The first step in developing a smart agent and getting a feel for the game is by running the standard script (random actions for both players) a couple of times and going through the game videos. Alternatively, you can play against a random bot (see next section).**

## How to play
1. Install the ```requirements.txt``` or ```dev-requirements.txt``` file.

2. Navigate to the ```play.py``` file.

3. (OPTIONAL) Connect Player 2 to a smart agent.

4. Run the ```play.py``` file.

The standard controls in the play.py file are:
```
W (UP)
S (DOWN)
A (LEFT)
D (RIGHT)
SPACE (BOMB)

NOTE: The Bomb function for the bot has been disabled to make it unable to blow itself up.
```

**Standard example: Player vs. Random Bot**
![](docs/player_vs_random.gif)

# soccer_offense

***an easy-modified gym environment for single agent RL***
***
A striker is getting to get the ball into the goal while a goalie is trying to
prevent it from happen.
***
- At start of the game, the striker and the ball randomly spawn at one side of
the filed while the goalie spawns randomly at the opposite side near the goal.
- If the ball is in certain range around striker, striker will carry the ball automatically
while moving and be able to shoot. the striker can choose to shoot at either left, middle
or right area of the goal, there is a small chance of missing shooting left or right, while
shooting at middle will not miss.
- The ball will be captured by goalie once the ball is close enough to the goalie,
and the game is over once captured. Goalie is controlled by environment with goalie_mode deciding
the behavior of the goalie.

- The game is over once:
    - the striker runs outside the field (bad ending)
    - the ball goes outside the field (bad ending)
    - the ball is captured by goalie (bad ending)
    - the striker achieves a goal (good ending)
***
The agent/human player plays the striker, the first reasonable thing to learn/do is to move
toward the ball and reach it, after that, the striker shall find his/her best chance to shoot
and achieve a goal.<br/><br/>
The friction slows the ball down, makes it possible that the ball stops before the goal,
and may forcing the striker to learn how to bypass the goalie to get closer to the goal.<br><br/>
The original game is extremely reward-sparse: only get one reward at the end of the game. you
can modify the reward under certain circumstances. See self._apply_rules for reward details.
***
## Installation
```bash
cd soccer-offense
pip install  -e .
```
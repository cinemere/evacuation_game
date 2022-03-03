# evacuation_game

RL implemetation of evacuation of one pedestrian with the help of one leader to a random exit.

---

## Brief description of RL agent:

### State (total 21 variables)

| Observative  | Format | Size | Decription |
| --- | --- | --- | --- |
| `wall collision`  | [up, down, right, left]  | *bool* x 4 | if/not collision in neighbouring __ cell |
| `agent direction`  | [up, down, right, left]  | *bool* x 4 | if/not agent in __ direction |
| `exit position` (rel)  | [up, down, right, left]  | *bool* x 4 | if/not exit in __ direction to agent |
| `pedestrian position` (rel)  | [up, down, right, left]  | *bool* x 4 | if/not pedestrian in __ direction to agent |
| `vision zone occupied` | 1 or 0 | *bool* | is pedestrian's vision zone occupied by agent |
| `agent-pedestrian distance` | [\|x1 - x2\|, \|y1 - y2\|] | *float* x 2 | distance between agent and pedestrian |
| `exit-pedestrian distance` | [\|x1 - x2\|, \|y1 - y2\|] | *float* x 2 | distance between exit and pedestrian |



![The simualtion after 1200 episodes](evacuation_game.gif)

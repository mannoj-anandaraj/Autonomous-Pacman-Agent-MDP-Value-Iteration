# mdpAgents.py
# parsons/20-nov-2017
#
# Version 1
#
# The starting point for CW2.
#
# Intended to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
# The agent here was written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util


class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print("Starting up MDPAgent!")
        # MDP data structures
        self.values = {}          # V(s)
        self.policy = {}          # pi(s)
        self.gamma = 0.88         # Discount factor

        # World model stored after registerInitialState
        self.walls = []
        self.width = 0
        self.height = 0

        name = "Pacman"

    # ----------------------------
    # HELPER: Extract world model
    def extractWorld(self, state):
        """
        Reads environment information using api.py
        to build the components needed for the MDP.
        """
        # Pacman position
        pacman = api.whereAmI(state)

        # Walls
        walls = api.walls(state)

        # Food
        food_list = api.food(state)

        # Ghosts and ghost states (scared or active)
        ghosts = api.ghosts(state)
        ghost_states = api.ghostStatesWithTimes(state)

        # Legal actions
        legal = api.legalActions(state)

        return {
            "pacman": pacman,
            "walls": walls,
            "food": food_list,
            "ghosts": ghosts,
            "ghost_states": ghost_states,
            "legal": legal
        }

    # ----------------------------
    # Compute next position given an action
    def nextPosition(self, pos, action):
        x, y = pos
        if action == Directions.NORTH:
            new_pos = (x, y + 1)
        elif action == Directions.SOUTH:
            new_pos = (x, y - 1)
        elif action == Directions.EAST:
            new_pos = (x + 1, y)
        elif action == Directions.WEST:
            new_pos = (x - 1, y)
        else:
            return pos  # STOP or unknown action

        # out-of-bounds
        if not (0 <= new_pos[0] < self.width and 0 <= new_pos[1] < self.height):
            return pos

        # wall check
        if new_pos in self.wall_coords:
            return pos

        return new_pos

    # ----------------------------
    # Transition model P(s' | s, a)
    # Deterministic: If wall, stay in place.
    def getTransitionStates(self, state_pos, action):
        next_pos = self.nextPosition(state_pos, action)

        # If next position is a wall, stay where you are
        if next_pos in self.wall_coords:
            next_pos = state_pos

        return [next_pos]   # deterministic

    # ----------------------------
    # Reward function R(s, a, s')
    def getReward(self, s, next_s, food, ghosts):
        # small step penalty
        reward = -1

        # if next position has food
        if next_s in food:
            reward += 18

        # Convert ghost positions to grid cells
        ghost_cells = [(int(g[0]), int(g[1])) for g in ghosts]

        # Collision with ghost is deadly
        if next_s in ghost_cells:
            reward -= 5000

        # Penalty for being close to a ghost (Manhattan distance of 1)
        for gx, gy in ghost_cells:
            if abs(next_s[0] - gx) + abs(next_s[1] - gy) == 1:
                reward -= 120

        # Ghost danger radius is 2 (small penalty)
        for gx, gy in ghost_cells:
            if abs(next_s[0] - gx) + abs(next_s[1] - gy) == 2:
                reward -= 12

        # Small shaping reward for moving toward food
        if len(food) > 0:
            dist_before = min(abs(s[0] - fx) + abs(s[1] - fy) for fx, fy in food)
            dist_after = min(abs(next_s[0] - fx) + abs(next_s[1] - fy) for fx, fy in food)
            if dist_after < dist_before:
                reward += 1

        return reward

    # ------------------------------
    # Return all possible actions
    def getAllActions(self):
        return [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    # ------------------------------
    # Compute Q-value for (state, action)
    def computeQValue(self, s, a, food, ghosts):
        total = 0
        transitions = self.getTransitionStates(s, a)
        for next_s in transitions:
            r = self.getReward(s, next_s, food, ghosts)
            total += r + self.gamma * self.values[next_s]
        return total

    # ----------------------------
    # VALUE ITERATION
    def runValueIteration(self, food, ghosts, iterations=20):
        for _ in range(iterations):
            new_values = util.Counter()
            for s in self.states:
                legal_actions = self.getLegalActionsAtState(s)
                if len(legal_actions) == 0:  # terminal or trapped
                    new_values[s] = 0
                    continue

                # Evaluate all actions
                q_values = []
                for a in legal_actions:
                    q = self.computeQValue(s, a, food, ghosts)
                    q_values.append(q)

                # Best Q-value becomes new V(s)
                new_values[s] = max(q_values)

            # Update values after one full sweep
            self.values = new_values

    # ----------------------------
    # Get best action from value function
    def getBestAction(self, pacman_pos, food, ghosts):
        legal_actions = self.getLegalActionsAtState(pacman_pos)
        if len(legal_actions) == 0:
            return Directions.STOP

        best_a = None
        best_q = float("-inf")
        for a in legal_actions:
            q = self.computeQValue(pacman_pos, a, food, ghosts)
            if q > best_q:
                best_q = q
                best_a = a
        return best_a

    # ------------------------------
    # Gets run after an MDPAgent object is created and once there is
    def registerInitialState(self, state):
        print("Running registerInitialState for MDPAgent!")
        print("I'm at:")
        print(api.whereAmI(state))

        # ---- GET TRUE WALL GRID (Grid object, NOT raw list) ----
        walls = state.getWalls()   # This is a Grid object with width/height stored correctly
        self.width = walls.width
        self.height = walls.height
        print("Correct walls dimensions:", self.width, "x", self.height)

        # Convert Grid into simple lookup set
        self.wall_coords = set()
        for x in range(self.width):
            for y in range(self.height):
                if walls[x][y]:
                    self.wall_coords.add((x, y))

        # Create list of valid (x,y) states
        self.states = []
        for x in range(self.width):
            for y in range(self.height):
                if (x, y) not in self.wall_coords:
                    self.states.append((x, y))

        print("Number of states:", len(self.states))

        # Initialise value function
        self.values = {s: 0.0 for s in self.states}

    # ------------------------------
    # Here run when the game ends
    def final(self, state):
        print("Looks like the game just ended!")

    # ------------------------------
    # Get legal actions at any state
    def getLegalActionsAtState(self, state):
        """
        Returns the legal actions from a given (x, y) state.
        Legal = any action that does not hit a wall or go outside the grid.
        """
        x, y = state
        legal = []

        # NORTH
        if y + 1 < self.height and (x, y + 1) not in self.wall_coords:
            legal.append(Directions.NORTH)

        # SOUTH
        if y - 1 >= 0 and (x, y - 1) not in self.wall_coords:
            legal.append(Directions.SOUTH)

        # EAST
        if x + 1 < self.width and (x + 1, y) not in self.wall_coords:
            legal.append(Directions.EAST)

        # WEST
        if x - 1 >= 0 and (x - 1, y) not in self.wall_coords:
            legal.append(Directions.WEST)

        # If no movement is possible (rare), allow STOP
        if not legal:
            legal.append(Directions.STOP)

        return legal

    # ------------------------------
    # This is the main function that gets called by pacman.py each time
    def getAction(self, state):
        # Extract necessary world information
        world = self.extractWorld(state)
        pacman = world["pacman"]
        food = world["food"]
        ghosts = world["ghosts"]   # raw ghosts

        # Run value iteration for the current world state
        self.runValueIteration(food, ghosts, iterations=20)

        # Choose the best action based on the updated value function
        best_action = self.getBestAction(pacman, food, ghosts)

        # Legal actions from API
        legal = api.legalActions(state)

        # If for some reason the best action isn't legal, fallback to a safe move
        if best_action not in legal:
            legal_no_stop = [a for a in legal if a != Directions.STOP]
            if legal_no_stop:
                return api.makeMove(random.choice(legal_no_stop), legal)
            return api.makeMove(random.choice(legal), legal)

        return api.makeMove(best_action, legal)

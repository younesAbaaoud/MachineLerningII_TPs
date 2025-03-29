import numpy as np
import random
import time
import tkinter as tk

# Environment settings
GRID_SIZE = 5
START_POS = (0, 0)

# Rewards
REWARD_TREASURE = 10
REWARD_TRAP = -10
REWARD_MOVE = -1

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPSILON = 0.1  # Exploration rate
EPISODES = 500

# Actions (Up, Down, Left, Right)
ACTIONS = {
    0: (-1, 0),  # Up
    1: (1, 0),   # Down
    2: (0, -1),  # Left
    3: (0, 1)    # Right
}

# Initialize Q-table
Q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))


# Check if a move is valid
def is_valid(pos):
    return 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE


# Generate random traps and treasure
def generate_random_environment():
    traps = set()
    while len(traps) < 3:  # Randomly place 3 traps
        trap_pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if trap_pos != START_POS:  # Ensure the trap is not at start
            traps.add(trap_pos)

    while True:
        treasure = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
        if treasure not in traps and treasure != START_POS:
            break  # Ensure the treasure does not overlap with traps or start position

    return list(traps), treasure


# Train Q-learning agent
def train_q_learning():
    global Q_table, TRAPS, TREASURE_POS
    TRAPS, TREASURE_POS = generate_random_environment()  # New environment each time

    for episode in range(EPISODES):
        state = START_POS
        while state != TREASURE_POS and state not in TRAPS:
            # Choose action (epsilon-greedy)
            if random.uniform(0, 1) < EPSILON:
                action = random.choice(list(ACTIONS.keys()))  # Explore
            else:
                action = np.argmax(Q_table[state])  # Exploit

            # Get next position
            next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])

            # Check if move is valid
            if not is_valid(next_state):
                next_state = state  # Stay in place

            # Determine reward
            if next_state == TREASURE_POS:
                reward = REWARD_TREASURE
            elif next_state in TRAPS:
                reward = REWARD_TRAP
            else:
                reward = REWARD_MOVE

            # Update Q-table
            Q_table[state][action] += ALPHA * (
                reward + GAMMA * np.max(Q_table[next_state]) - Q_table[state][action]
            )

            # Move to next state
            state = next_state


# Tkinter UI
class QLearningUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reinforcement Learning Agent")

        self.canvas = tk.Canvas(root, width=500, height=500)
        self.canvas.pack()

        self.cell_size = 500 // GRID_SIZE
        self.agent_pos = START_POS
        self.draw_grid()

        # Start button
        self.start_button = tk.Button(root, text="Start Simulation", command=self.start_simulation)
        self.start_button.pack()

    def draw_grid(self):
        """Draws the grid, traps, and treasure"""
        self.canvas.delete("all")

        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x0, y0 = j * self.cell_size, i * self.cell_size
                x1, y1 = x0 + self.cell_size, y0 + self.cell_size

                # Draw cells
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")

                # Draw traps
                if (i, j) in TRAPS:
                    self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="red")

                # Draw treasure
                if (i, j) == TREASURE_POS:
                    self.canvas.create_oval(x0 + 10, y0 + 10, x1 - 10, y1 - 10, fill="gold")

        # Draw agent
        self.draw_agent()

    def draw_agent(self):
        """Draws the agent in the current position"""
        x0, y0 = self.agent_pos[1] * self.cell_size, self.agent_pos[0] * self.cell_size
        x1, y1 = x0 + self.cell_size, y0 + self.cell_size
        self.canvas.create_oval(x0 + 15, y0 + 15, x1 - 15, y1 - 15, fill="blue", tags="agent")

    def start_simulation(self):
        """Runs the trained agent step by step"""
        global TRAPS, TREASURE_POS
        train_q_learning()  # Generate a new environment and train the agent
        state = START_POS

        while state != TREASURE_POS and state not in TRAPS:
            self.agent_pos = state
            self.draw_grid()
            self.root.update()
            time.sleep(0.5)  # Delay to visualize movement

            # Choose the best action
            action = np.argmax(Q_table[state])
            next_state = (state[0] + ACTIONS[action][0], state[1] + ACTIONS[action][1])

            if not is_valid(next_state):
                next_state = state  # Stay in place

            state = next_state


# Run the program
if __name__ == "__main__":
    train_q_learning()  # Train the agent
    root = tk.Tk()
    app = QLearningUI(root)
    root.mainloop()

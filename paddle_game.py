import pygame
import random
import numpy as np
import sys

# Initialize Pygame
pygame.init()

# Game Parameters
grid_size = 11
cell_size = 40
screen_size = (grid_size * cell_size, grid_size * cell_size)
background_color = (30, 30, 30)
paddle_color = (255, 255, 255)
ball_color = (255, 0, 0)
fps = 10

# Set up the display
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Paddle Game with SARSA and Q-Learning")

# Paddle and Ball Parameters
points = 0
paddle_pos = grid_size // 2
ball_pos = [0, random.randint(0, grid_size - 1)]

# RL Parameters
alpha = 0.1    # Learning rate
gamma = 0.9    # Discount factor
epsilon = 0.1  # Exploration rate
q_table = np.zeros((grid_size, grid_size, 3))  # Q-table for both algorithms
actions = [-1, 0, 1]  # Actions: move left, stay, move right

# Modes
MODE = 'SARSA'          # Choose between 'SARSA' and 'Q-Learning'
MANUAL_MODE = True      # Toggle for manual control

# Functions
def reset_ball():
    """ Reset the ball to the top row at a random horizontal position """
    ball_pos[0] = 0
    ball_pos[1] = random.randint(0, grid_size - 1)

def choose_action(state):
    """ Choose an action based on epsilon-greedy policy """
    if random.uniform(0, 1) < epsilon:
        return random.choice(actions)
    return actions[np.argmax(q_table[state[0], state[1]])]

def get_reward(paddle_pos, ball_pos):
    """ Reward function: +1 for catching the ball, -1 for missing """
    if ball_pos[0] == grid_size - 1:
        return 1 if paddle_pos == ball_pos[1] else -1
    return 0

# Game Loop
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(background_color)

    # Current state and action (used only if in AI mode)
    if not MANUAL_MODE:
        state = (paddle_pos, ball_pos[1])
        action_index = actions.index(choose_action(state))
        action = actions[action_index]
    else:
        action = 0  # No automatic action in manual mode

    # Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if MANUAL_MODE:
                if event.key == pygame.K_LEFT:
                    action = -1  # Move paddle left
                elif event.key == pygame.K_RIGHT:
                    action = 1   # Move paddle right

    # Update Paddle Position
    paddle_pos = max(0, min(grid_size - 1, paddle_pos + action))

    # Update Ball Position
    ball_pos[0] += 1  # Move ball downward

    # Check for terminal state and get reward
    reward = get_reward(paddle_pos, ball_pos)
    points += reward if reward else 0
    done = ball_pos[0] == grid_size - 1

    # Next state and update for AI
    if not MANUAL_MODE:
        next_state = (paddle_pos, ball_pos[1])
        next_action_index = actions.index(choose_action(next_state))
        next_action = actions[next_action_index]

        # Update Q-values based on chosen mode
        if MODE == 'Q-Learning':
            # Q-Learning update
            q_table[state[0], state[1], action_index] += alpha * (reward + gamma * np.max(q_table[next_state[0], next_state[1]]) - q_table[state[0], state[1], action_index])
        elif MODE == 'SARSA':
            # SARSA update
            q_table[state[0], state[1], action_index] += alpha * (reward + gamma * q_table[next_state[0], next_state[1], next_action_index] - q_table[state[0], state[1], action_index])

    # Reset ball if it reaches the bottom
    if done:
        reset_ball()

    # Draw Paddle
    paddle_rect = pygame.Rect(paddle_pos * cell_size, (grid_size - 1) * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, paddle_color, paddle_rect)

    # Draw Ball
    ball_rect = pygame.Rect(ball_pos[1] * cell_size, ball_pos[0] * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, ball_color, ball_rect)

    # Display Points
    font = pygame.font.Font(None, 36)
    text = font.render(f"Points: {points}", True, (255, 255, 255))
    screen.blit(text, (10, 10))

    # Update the display
    pygame.display.flip()
    clock.tick(fps)  # Control game speed

# Quit the game
pygame.quit()
sys.exit()

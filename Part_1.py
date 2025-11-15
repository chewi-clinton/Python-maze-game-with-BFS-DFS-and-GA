import pygame
import sys

pygame.init()

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)

CELL_SIZE = 40
FPS = 60

MAZE_1 = [
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

MAZE_2 = [
    [0, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1]
]


class Maze:
    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.rows = len(grid)
        self.cols = len(grid[0])
    
    def is_valid_move(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] == 0
        return False
    
    def get_neighbors(self, row, col):
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_move(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors


class Player:
    def __init__(self, start_pos, cell_size):
        self.row, self.col = start_pos
        self.start_pos = start_pos
        self.image = self.create_player_image(cell_size)
    
    def create_player_image(self, cell_size):

        image = pygame.image.load("player.png")
       
        size = int(cell_size * 1.0)
        return pygame.transform.scale(image, (size, size))
    
    def move(self, direction, maze):
        new_row, new_col = self.row, self.col
        
        if direction == "UP":
            new_row -= 1
        elif direction == "DOWN":
            new_row += 1
        elif direction == "LEFT":
            new_col -= 1
        elif direction == "RIGHT":
            new_col += 1
        
        if maze.is_valid_move(new_row, new_col):
            self.row, self.col = new_row, new_col
            return True
        return False
    
    def reset(self):
        self.row, self.col = self.start_pos
    
    def get_position(self):
        return (self.row, self.col)


def draw_maze(screen, maze):
    for row in range(maze.rows):
        for col in range(maze.cols):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            
            if maze.grid[row][col] == 1:
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE))
            else:
                pygame.draw.rect(screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
            
            pygame.draw.rect(screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE), 1)


def draw_start_position(screen, maze):
    row, col = maze.start
    center_x = col * CELL_SIZE + CELL_SIZE // 2
    center_y = row * CELL_SIZE + CELL_SIZE // 2
    pygame.draw.circle(screen, LIGHT_GREEN, (center_x, center_y), CELL_SIZE // 3)
    pygame.draw.circle(screen, GREEN, (center_x, center_y), CELL_SIZE // 3, 3)


def draw_goal_position(screen, maze):
    row, col = maze.goal
    x = col * CELL_SIZE
    y = row * CELL_SIZE
    
    checker_size = CELL_SIZE // 4
    for i in range(4):
        for j in range(4):
            if (i + j) % 2 == 0:
                pygame.draw.rect(screen, RED, 
                               (x + i * checker_size, y + j * checker_size, 
                                checker_size, checker_size))
            else:
                pygame.draw.rect(screen, DARK_RED, 
                               (x + i * checker_size, y + j * checker_size, 
                                checker_size, checker_size))
    
    pygame.draw.rect(screen, RED, (x, y, CELL_SIZE, CELL_SIZE), 3)


def draw_player(screen, player):
    x = player.col * CELL_SIZE + (CELL_SIZE - player.image.get_width()) // 2
    y = player.row * CELL_SIZE + (CELL_SIZE - player.image.get_height()) // 2
    screen.blit(player.image, (x, y))


def draw_info_panel(screen, player, maze, window_width, window_height):
    font = pygame.font.Font(None, 24)
    
    if player.get_position() == maze.goal:
        text = font.render(" YOU WIN! Press R to restart", True, GREEN)
    else:
        text = font.render("Use Arrow Keys to Move | R: Restart | Q: Quit", True, BLACK)
    
    text_rect = text.get_rect(center=(window_width // 2, window_height - 15))
    bg_rect = pygame.Rect(0, window_height - 30, window_width, 30)
    pygame.draw.rect(screen, LIGHT_GREEN if player.get_position() == maze.goal else WHITE, bg_rect)
    pygame.draw.rect(screen, BLACK, bg_rect, 2)
    screen.blit(text, text_rect)


def main():
    current_maze_grid = MAZE_2
    start_position = (0, 0)
    goal_position = (5, 5)
    
    maze = Maze(current_maze_grid, start_position, goal_position)
    player = Player(start_position, CELL_SIZE)
    
    WINDOW_WIDTH = maze.cols * CELL_SIZE
    WINDOW_HEIGHT = maze.rows * CELL_SIZE + 30
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Yxng´s Maze Game - Part 1")
    clock = pygame.time.Clock()
    
    running = True
    won = False
    
    print("=" * 50)
    print("MAZE GAME - Arrow Keys: Move | R: Restart | Q: Quit")
    print("=" * 50)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.key == pygame.K_r:
                    player.reset()
                    won = False
                    print("Game restarted!")
                
                elif not won:
                    moved = False
                    if event.key == pygame.K_UP:
                        moved = player.move("UP", maze)
                    elif event.key == pygame.K_DOWN:
                        moved = player.move("DOWN", maze)
                    elif event.key == pygame.K_LEFT:
                        moved = player.move("LEFT", maze)
                    elif event.key == pygame.K_RIGHT:
                        moved = player.move("RIGHT", maze)
                    
                    if moved and player.get_position() == maze.goal:
                        won = True
                        print("You reached the goal!")
        
        screen.fill(WHITE)
        draw_maze(screen, maze)
        draw_start_position(screen, maze)
        draw_goal_position(screen, maze)
        draw_player(screen, player)
        draw_info_panel(screen, player, maze, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
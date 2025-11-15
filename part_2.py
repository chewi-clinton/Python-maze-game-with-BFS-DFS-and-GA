import pygame
import sys
from collections import deque
import time

pygame.init()

#  CONSTANTS 
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GREEN = (144, 238, 144)
DARK_RED = (139, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (147, 112, 219)
LIGHT_BLUE = (173, 216, 230)

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


# MAZE CLASS 
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


#  PLAYER CLASS 
class Player:
    def __init__(self, start_pos, cell_size):
        self.row, self.col = start_pos
        self.start_pos = start_pos
        self.image = self.create_player_image(cell_size)
    
    def create_player_image(self, cell_size):
        try:
            image = pygame.image.load("player.png")
            size = int(cell_size * 1.0)
            return pygame.transform.scale(image, (size, size))
        except:
         
            size = int(cell_size * 0.7)
            surface = pygame.Surface((size, size), pygame.SRCALPHA)
            center = size // 2
            pygame.draw.circle(surface, (255, 220, 177), (center, center - 5), size // 4)
            pygame.draw.ellipse(surface, (70, 130, 180), 
                              (center - size//6, center + 3, size//3, size//2))
            pygame.draw.circle(surface, BLACK, (center - 4, center - 8), 2)
            pygame.draw.circle(surface, BLACK, (center + 4, center - 8), 2)
            pygame.draw.arc(surface, BLACK, (center - 6, center - 8, 12, 10), 3.14, 6.28, 2)
            return surface
    
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


#SEARCH ALGORITHMS 
def dfs(maze, start, goal):
    """Depth-First Search"""
    visited = set()
    visited_order = []
    path = []
    
    def dfs_helper(current):
        row, col = current
        visited.add(current)
        visited_order.append(current)
        path.append(current)
        
        if current == goal:
            return True
        
        for neighbor in maze.get_neighbors(row, col):
            if neighbor not in visited:
                if dfs_helper(neighbor):
                    return True
        
        path.pop()
        return False
    
    success = dfs_helper(start)
    return (path if success else None, visited_order, len(visited_order))


def bfs(maze, start, goal):
    """Breadth-First Search"""
    queue = deque([start])
    visited = {start}
    visited_order = [start]
    parent = {start: None}
    
    while queue:
        current = queue.popleft()
        
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            path.reverse()
            return (path, visited_order, len(visited_order))
        
        row, col = current
        for neighbor in maze.get_neighbors(row, col):
            if neighbor not in visited:
                visited.add(neighbor)
                visited_order.append(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)
    
    return (None, visited_order, len(visited_order))


#  DRAWING FUNCTIONS 
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


def draw_cell(screen, cell, color, alpha=255):
    """Draw colored cell for visualization"""
    row, col = cell
    x = col * CELL_SIZE + 2
    y = row * CELL_SIZE + 2
    size = CELL_SIZE - 4
    
    if alpha < 255:
        s = pygame.Surface((size, size))
        s.set_alpha(alpha)
        s.fill(color)
        screen.blit(s, (x, y))
    else:
        pygame.draw.rect(screen, color, (x, y, size, size))


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


def draw_info_panel(screen, text, window_width, window_height, color=BLACK):
    font = pygame.font.Font(None, 20)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect(center=(window_width // 2, window_height - 15))
    bg_rect = pygame.Rect(0, window_height - 30, window_width, 30)
    pygame.draw.rect(screen, WHITE, bg_rect)
    pygame.draw.rect(screen, BLACK, bg_rect, 2)
    screen.blit(text_surface, text_rect)


# VISUALIZATION 
def visualize_search(screen, maze, visited_order, final_path, algorithm_name, delay=30):
    """Animate search process"""
    clock = pygame.time.Clock()
    
    # Animate exploration
    for i, cell in enumerate(visited_order):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        screen.fill(WHITE)
        draw_maze(screen, maze)
        
        # Draw explored cells
        for visited_cell in visited_order[:i+1]:
            if visited_cell != maze.start and visited_cell != maze.goal:
                draw_cell(screen, visited_cell, YELLOW, alpha=150)
        
        draw_start_position(screen, maze)
        draw_goal_position(screen, maze)
        
        draw_info_panel(screen, f"{algorithm_name} - Exploring... ({i+1}/{len(visited_order)})", 
                       maze.cols * CELL_SIZE, maze.rows * CELL_SIZE + 30)
        
        pygame.display.flip()
        pygame.time.delay(delay)
    
    # Show final path
    if final_path:
        screen.fill(WHITE)
        draw_maze(screen, maze)
        
        # Draw all explored (light blue)
        for visited_cell in visited_order:
            if visited_cell != maze.start and visited_cell != maze.goal:
                draw_cell(screen, visited_cell, LIGHT_BLUE, alpha=100)
        
        # Draw final path (purple)
        for path_cell in final_path:
            if path_cell != maze.start and path_cell != maze.goal:
                draw_cell(screen, path_cell, PURPLE)
        
        draw_start_position(screen, maze)
        draw_goal_position(screen, maze)
        
        draw_info_panel(screen, 
                       f"{algorithm_name} Complete! Path: {len(final_path)} | Explored: {len(visited_order)} | Press any key", 
                       maze.cols * CELL_SIZE, maze.rows * CELL_SIZE + 30, GREEN)
        
        pygame.display.flip()
        
        # Wait for key press
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting = False
    else:
        draw_info_panel(screen, f"{algorithm_name} - No path found! Press any key", 
                       maze.cols * CELL_SIZE, maze.rows * CELL_SIZE + 30, RED)
        pygame.display.flip()
        
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    waiting = False


#  MAIN GAME 
def main():
    current_maze_grid = MAZE_2
    start_position = (0, 0)
    goal_position = (5, 5)
    
    maze = Maze(current_maze_grid, start_position, goal_position)
    player = Player(start_position, CELL_SIZE)
    
    WINDOW_WIDTH = maze.cols * CELL_SIZE
    WINDOW_HEIGHT = maze.rows * CELL_SIZE + 30
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Yxng's Maze Game - Part 2 (DFS & BFS)")
    clock = pygame.time.Clock()
    
    running = True
    won = False
    
    print("=" * 60)
    print("MAZE GAME CONTROLS")
    print("=" * 60)
    print("Arrow Keys: Manual movement")
    print("D: Run DFS (Depth-First Search)")
    print("B: Run BFS (Breadth-First Search)")
    print("R: Restart | Q: Quit")
    print("=" * 60)
    
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
                    print("\nGame restarted!")
                
                # DFS
                elif event.key == pygame.K_d:
                    print("\n--- Running DFS ---")
                    start_time = time.time()
                    path, visited, nodes = dfs(maze, maze.start, maze.goal)
                    duration = time.time() - start_time
                    
                    if path:
                        print(f"✓ Path found!")
                        print(f"  Path length: {len(path)}")
                        print(f"  Nodes explored: {nodes}")
                        print(f"  Time: {duration:.3f}s")
                    else:
                        print("✗ No path found!")
                    
                    visualize_search(screen, maze, visited, path, "DFS")
                
                # BFS
                elif event.key == pygame.K_b:
                    print("\n--- Running BFS ---")
                    start_time = time.time()
                    path, visited, nodes = bfs(maze, maze.start, maze.goal)
                    duration = time.time() - start_time
                    
                    if path:
                        print(f"✓ Path found!")
                        print(f"  Path length: {len(path)}")
                        print(f"  Nodes explored: {nodes}")
                        print(f"  Time: {duration:.3f}s")
                    else:
                        print("✗ No path found!")
                    
                    visualize_search(screen, maze, visited, path, "BFS")
                
                # Manual play
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
                        print("\n You reached the goal manually!")
        
        screen.fill(WHITE)
        draw_maze(screen, maze)
        draw_start_position(screen, maze)
        draw_goal_position(screen, maze)
        draw_player(screen, player)
        
        if won:
            draw_info_panel(screen, " YOU WIN! Press R to restart", 
                          WINDOW_WIDTH, WINDOW_HEIGHT, GREEN)
        else:
            draw_info_panel(screen, "Arrows: Move | D: DFS | B: BFS | R: Restart | Q: Quit", 
                          WINDOW_WIDTH, WINDOW_HEIGHT)
        
        pygame.display.flip()
        clock.tick(FPS)
    
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
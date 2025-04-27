import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import config as cfg

pygame.init()
font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 20)

# Load configuration values
GREEN1 = cfg.COLOR_GREEN1
GREEN2 = cfg.COLOR_GREEN2
BG1 = cfg.COLOR_BG1
BG2 = cfg.COLOR_BG2
LB = cfg.COLOR_LIGHT_BORDER

BLOCK_SIZE = cfg.BLOCK_SIZE
SPEED = cfg.SPEED
BAR_HEIGHT = cfg.BAR_HEIGHT

class Direction(Enum):
    """Enumeration for possible snake directions."""
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SmartSnake:
    """Class representing the Snake game environment."""
    def __init__(self, w=640, h=480):
        """Initialize the game environment."""
        self.w = w
        self.h = h
        self.apple_image = pygame.image.load('resources/apple.webp') 
        self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))

        self.record = 0  # Current session record
        self.new_record_flash = 0 # Frames left for "NEW RECORD" flashing
        self.shine_frames = 0 # Frames left for snake shining when eating apple

        self.display = pygame.display.set_mode((self.w, self.h + BAR_HEIGHT))
        pygame.display.set_caption('Smart Snake by Trinidad Monreal')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game state to start a new game."""
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.apple = None
        self.place_apple()
        self.frame_iteration = 0


    def place_apple(self):
        """Place a new apple randomly on the board, avoiding the snake's body."""
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.apple = Point(x, y)
        if self.apple in self.snake:
            self.place_apple()


    def play_step(self, action):
        """
        Play one frame of the game given an action.
        
        Returns:
            reward (int): reward earned this step
            game_over (bool): whether the game is over
            score (int): current score
        """
        self.frame_iteration += 1
        # Handle quit events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move snake
        self.action_move(action) # update the head
        self.snake.insert(0, self.head)
        
        # Check for collisions
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if apple was eaten
        if self.head == self.apple:
            self.score += 1
            reward = 10
            self.place_apple()
            self.shine_frames = 5 
        else:
            self.snake.pop()
        
        # Update visuals
        self.draw_screen()
        self.clock.tick(SPEED)

        # 6. Return game over and score
        return reward, game_over, self.score
    
    def set_record(self, new_record):
        """Update the session record if a new one is achieved."""
        if new_record > self.record:
            self.record = new_record
            self.new_record_flash = 30  

    def is_collision(self, pt=None):
        """
        Check if the snake collides with wall or itself.
        
        Args:
            pt (Point, optional): Point to check collision at. Defaults to head.

        Returns:
            bool: True if collision occurs, False otherwise.
        """
        if pt is None:
            pt = self.head
        # Check wall collisions
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check collisions with itself
        if pt in self.snake[1:]:
            return True

        return False

    def draw_screen(self):       
        """Draw all elements (background, snake, apple, score bar) on the game window."""
        # Draw Score and Record
        self.draw_score_bar()

        # Draw checkerboard background
        for y in range(0, self.h, BLOCK_SIZE):
            for x in range(0, self.w, BLOCK_SIZE):
                color = BG1 if (x // BLOCK_SIZE + y // BLOCK_SIZE) % 2 == 0 else BG2
                pygame.draw.rect(self.display, color, pygame.Rect(x, y + BAR_HEIGHT, BLOCK_SIZE, BLOCK_SIZE))

        # Draw snake
        snake_color1 = (124, 252, 0) if self.shine_frames > 0 else GREEN1
        snake_color2 = (144, 238, 144) if self.shine_frames > 0 else GREEN2

        for pt in self.snake:
            pygame.draw.rect(self.display, snake_color1, pygame.Rect(pt.x, pt.y + BAR_HEIGHT, BLOCK_SIZE, BLOCK_SIZE), border_radius=5)
            pygame.draw.rect(self.display, snake_color2, pygame.Rect(pt.x+4, pt.y+4 + BAR_HEIGHT, 12, 12), border_radius=5)

        # Draw apple
        self.display.blit(self.apple_image, (self.apple.x, self.apple.y + BAR_HEIGHT))

        # Show flashing "NEW RECORD!!" if necessary
        if self.new_record_flash > 0:
            big_font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 40)
            big_font.set_bold(True)
            alpha = min(255, (30 - self.new_record_flash) * 8)  
            
            # Render text to a new surface
            flash_surface = big_font.render('¡NEW RECORD!', True, (255, 215, 0))
            flash_surface.set_alpha(alpha)
            
            # Create a temporary surface with per-pixel alpha
            temp_surface = pygame.Surface((self.w, self.h + BAR_HEIGHT), pygame.SRCALPHA)
            text_rect = flash_surface.get_rect(center=(self.w//2, self.h//2))
            temp_surface.blit(flash_surface, text_rect)
            
            # Blit temporary surface over display
            self.display.blit(temp_surface, (0,0))

            self.new_record_flash -= 1

        if self.shine_frames > 0:
            self.shine_frames -= 1

        # Draw border around game area
        border_thickness = 2
        pygame.draw.rect(self.display, LB,
                        pygame.Rect(0, BAR_HEIGHT, self.w, self.h),
                        border_thickness, border_radius=5)

        pygame.display.flip()

    def draw_score_bar(self):
        """Draw the top bar showing the current score and record."""
        pygame.draw.rect(self.display, BG1, pygame.Rect(0, 0, self.w, BAR_HEIGHT))
        pygame.draw.line(self.display, (255, 255, 255), (0, BAR_HEIGHT), (self.w, BAR_HEIGHT), 2)

        font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 20)
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        record_text = font.render(f"Record: {self.record}", True, (255, 255, 255))

        self.display.blit(score_text, (10, 10))
        self.display.blit(record_text, (self.w - record_text.get_width() - 10, 10))

    def set_session_record(self, session_record):
        """Set the session record manually (used when loading from a pretrained model)."""
        self.record = session_record 

    def flash_new_record(self):
        """Trigger the flash animation for setting a new global record."""
        self.new_record_flash = 30  

    def action_move(self, action):
        """
        Update snake's direction based on the action.
        
        Args:
            action (list): [straight, right, left] move encoding.
        """

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # No change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # Right turn 
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # Left turn 

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 20)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
GREEN1 = (34, 139, 34)   # Darker 
GREEN2 = (50, 205, 50)   # Lighter 
BG1 = (30, 30, 60)       # Darker blue tile
BG2 = (50, 50, 80)       # Lighter blue tile
LB = (100, 130, 160)  # Light blue border

BLOCK_SIZE = 20
SPEED = 40
BAR_HEIGHT = 40

class SnakeGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.apple_image = pygame.image.load('resources/apple.webp') 
        self.apple_image = pygame.transform.scale(self.apple_image, (BLOCK_SIZE, BLOCK_SIZE))

        self.record = 0  
        self.new_record_flash = 0
        self.shine_frames = 0

        self.display = pygame.display.set_mode((self.w, self.h + BAR_HEIGHT))
        pygame.display.set_caption('Smart Snake by Trinidad Monreal')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.shine_frames = 5 
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
    def set_record(self, new_record):
        if new_record > self.record:
            self.record = new_record
            self.new_record_flash = 30  


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):       
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

        # Draw food (apple)
        self.display.blit(self.apple_image, (self.food.x, self.food.y + BAR_HEIGHT))


        # Show "NEW RECORD!!" if flashing
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

        border_thickness = 2
        pygame.draw.rect(self.display, LB,
                        pygame.Rect(0, BAR_HEIGHT, self.w, self.h),
                        border_thickness, border_radius=5)

        pygame.display.flip()

    def draw_score_bar(self):
        # Draw bar background
        pygame.draw.rect(self.display, BG1, pygame.Rect(0, 0, self.w, BAR_HEIGHT))
        pygame.draw.line(self.display, (255, 255, 255), (0, BAR_HEIGHT), (self.w, BAR_HEIGHT), 2)

        # Load font
        font = pygame.font.Font('resources/PressStart2P-Regular.ttf', 20)

        # Render Score and Record
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        record_text = font.render(f"Record: {self.record}", True, (255, 255, 255))

        # Blit texts
        self.display.blit(score_text, (10, 10))
        self.display.blit(record_text, (self.w - record_text.get_width() - 10, 10))

    def set_session_record(self, session_record):
        self.record = session_record 

    def flash_new_record(self):
        self.new_record_flash = 30  

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

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
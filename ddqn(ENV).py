
# env_tetris.py
# Tetris environment (environment-only; no agent or training logic)
from random import randrange as rand
import pygame
import sys
import copy
import numpy as np

# The configuration
cell_size = 60
cols = 7
rows = 14
maxfps = 30

boundary_size = 40

colors = [
    (0, 0, 0),
    (255, 150, 150),
    (100, 200, 115),
    (120, 108, 245),
    (255, 140, 50),
    (50, 120, 52),
    (146, 202, 73),
    (150, 161, 218),
    (35, 35, 35)  # Helper color for background grid
]

# Define the shapes of the single parts
tetris_shapes = [

    [[1, 1, 1, 1]],

    [[2, 0],
     [2, 2],
     [2, 0]],

    [[3, 3],
     [3, 3]],

    [[4, 0],
     [4, 4],
     [0, 4]],

    [[5, 0],
     [5, 0],
     [5, 5]],

    [[0, 6],
     [6, 6],
     [6, 0]],

    [[0, 7],
     [0, 7],
     [7, 7]]
]
game_score = 0


def rotate(l, n):
    """Rotate a list l by n (helper used for shape order)."""
    return l[n:] + l[:n]


def rotate_clockwise(shape):
    """Rotate a matrix (shape) clockwise by 90 degrees."""
    return [[shape[y][x]
             for y in range(len(shape))]
            for x in range(len(shape[0]) - 1, -1, -1)]


def check_collision(board, shape, offset):
    """
    Check if shape placed at offset (off_x, off_y) collides with board
    or is out-of-bounds. Returns True if collision detected.
    """
    off_x, off_y = offset
    for cy, row in enumerate(shape):
        for cx, cell in enumerate(row):
            try:
                if cell and board[cy + off_y][cx + off_x]:
                    return True
            except IndexError:
                return True
    return False


def remove_row(board, row):
    """Remove a full row from board and add empty row at top."""
    del board[row]
    return [[0 for i in range(cols)]] + board


def join_matrixes(mat1, mat2, mat2_off):
    """Join mat2 into mat1 at offset mat2_off (used when a stone lands)."""
    off_x, off_y = mat2_off
    for cy, row in enumerate(mat2):
        for cx, val in enumerate(row):
            mat1[cy + off_y - 1][cx + off_x] += val
    return mat1


def new_board():
    """Create a new board (rows x cols) and an extra floor row of 1s."""
    board = [[0 for x in range(cols)]
             for y in range(rows)]
    board += [[1 for x in range(cols)]]
    return board


class TetrisApp(object):
    """
    Tetris environment class.

    Key methods:
      - init_game(), new_stone(), move(), hard_drop(), drop(), step(action),
        reset(), run() (manual play)
    """

    def __init__(self, render=False):
        pygame.init()
        pygame.key.set_repeat(250, 25)
        self.render = render
        self.action_space = ['LEFT', 'RIGHT', 'DOWN', 'UP']
        self.action_size = len(self.action_space)
        self.color = ["red", "blue", "green", "yellow", "purple"]
        self.block_kind = len(self.color)

        # Screen setup
        self.width = cell_size * (cols + 6)
        self.gameWidth = cell_size * cols
        self.height = cell_size * rows
        self.rlim = cell_size * cols
        self.bground_grid = [[0 for x in range(cols)] for y in range(rows)]

        self.gameover = False
        self.paused = False

        self.default_font = pygame.font.Font(pygame.font.get_default_font(), 12)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.event.set_blocked(pygame.MOUSEMOTION)  # We do not need mouse motion events

        # Stone/shape bookkeeping
        self.stone_num = 0
        self.shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = [0, 1, 2, 3, 4, 5, 6]
        self.new_stone_flag = False

        # Initialize game state
        self.init_game()

    def set_name(self, a):
        pygame.display.set_caption("{}".format(a))

    def new_stone(self):
        """Generate/assign new stone and check immediate collision (gameover)."""
        self.new_stone_flag = True
        self.stone = self.next_stone[:]
        self.stone_num += 1

        if len(self.shapes) == 0:
            self.fix_shapes = rotate(self.fix_shapes, (rand(4) * 2))
            self.shapes = copy.deepcopy(self.fix_shapes)

        self.next_stone = tetris_shapes[self.shapes.pop(len(self.shapes) - 1)]
        self.stone_x = int(cols / 2 - len(self.stone[0]) / 2)
        if self.stone[0][0] == 1:
            self.stone_x = int(cols / 2 - (len(self.stone[0]) - 1) / 2)
        self.stone_y = 0

        if check_collision(self.board, self.stone, (self.stone_x, self.stone_y)):
            self.gameover = True

    def stone_number(self, stone):
        """Return value identifying a stone shape (helper used elsewhere)."""
        if stone[0][0] > 0:
            return stone[0][0]
        else:
            return stone[0][1]

    def reset(self):
        """Reset the game state and return the initial state (board)."""
        self.init_game()
        self.gameover = False
        return self.board

    def init_game(self):
        """Initialize/reset game internal variables and spawn first stone."""
        self.board = new_board()
        self.shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = [0, 1, 2, 3, 4, 5, 6]
        self.fix_shapes = rotate(self.fix_shapes, (rand(4) * 2))
        self.shapes = copy.deepcopy(self.fix_shapes)

        self.next_stone = tetris_shapes[self.shapes.pop(len(self.shapes) - 1)]
        self.hole = 0
        self.new_stone()
        self.level = 1
        self.score = 0
        self.lines = 0
        self.game_clline = 0
        self.total_clline = 0

        # Prepare initial display board state with current stone overlaid
        board_screen = copy.deepcopy(self.board)
        stone_m = len(self.stone)
        stone_n = len(self.stone[0])
        for m in range(stone_m):
            for n in range(stone_n):
                if self.stone[m][n] != 0:
                    board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]

        self.gameScreen = board_screen
        global game_score
        game_score = 0
        self.score_flag = False
        self.combo_score_flag = False
        self.allclear_score_flag = False
        self.combo_count = 0
        self.block_after_score = False
        self.hole_score = 0
        self.minus_score = 0
        self.plus_score = 0
        self.floor = 0

    def disp_msg(self, msg, topleft):
        """Render a message at the given top-left coordinates on the pygame screen."""
        x, y = topleft
        for line in msg.splitlines():
            self.screen.blit(
                self.default_font.render(line, False, (255, 255, 255), (0, 0, 0)),
                (x, y))
            y += 14

    def center_msg(self, msg):
        """Center a multi-line message on the game screen."""
        for i, line in enumerate(msg.splitlines()):
            msg_image = self.default_font.render(line, False, (255, 255, 255), (0, 0, 0))
            msgim_center_x, msgim_center_y = msg_image.get_size()
            msgim_center_x //= 2
            msgim_center_y //= 2
            self.screen.blit(msg_image, (
                self.width // 2 - msgim_center_x,
                self.height // 2 - msgim_center_y + i * 22))

    def draw_matrix(self, matrix, offset):
        """Draw a matrix (board or piece) on the pygame screen at given offset."""
        off_x, off_y = offset
        for y, row in enumerate(matrix):
            for x, val in enumerate(row):
                if val:
                    pygame.draw.rect(
                        self.screen,
                        colors[val],
                        pygame.Rect(
                            (off_x + x) * cell_size,
                            (off_y + y) * cell_size,
                            cell_size,
                            cell_size), 0)

    def add_cl_lines(self, n, floor):
        """Update scores and level after clearing lines (n: number of lines, floor: floor index)."""
        linescores = [0, 1, 3, 6, 10]
        floorscores = [0, 1, 1.17, 1.33, 1.5, 1.7, 1.8, 2, 2.17, 2.3, 2.5, 2.67, 2.8, 3]
        self.lines += n
        global game_score
        game_score += linescores[n]
        game_score += floorscores[floor]
        if linescores[n] > 0:
            self.score_flag = True
        if floorscores[floor] > 0:
            self.score_flage = True
        else:
            self.score_flag = False
        if self.lines >= self.level * 6:
            self.level += 1

    def move(self, delta_x):
        """Move current stone horizontally by delta_x if no collision."""
        if not self.gameover:
            new_x = self.stone_x + delta_x
            if new_x < 0:
                new_x = 0
            if new_x > cols - len(self.stone[0]):
                new_x = cols - len(self.stone[0])
            if not check_collision(self.board, self.stone, (new_x, self.stone_y)):
                self.stone_x = new_x
        return False

    def move_drop(self, n):
        """Move horizontally then hard-drop piece (used by action decoding)."""
        self.move(n)
        self.hard_drop(True)

    def quit(self):
        """Quit the pygame window and exit."""
        self.center_msg("Exiting...")
        pygame.display.update()
        sys.exit()

    def drop(self, manual):
        """Soft drop: move piece by one row; if collision, join piece to board, spawn new piece."""
        new_y = self.stone_y
        if not self.gameover:
            new_y = self.stone_y + 1
            cleared_rows = 0
            if check_collision(self.board, self.stone, (self.stone_x, new_y)):
                self.board = join_matrixes(self.board, self.stone, (self.stone_x, new_y))

                if new_y < rows / 2:
                    self.minus_score = (1 - new_y / 10) / 20

                self.new_stone()
                floor2 = []
                while True:
                    for i, row in enumerate(self.board[:-1]):
                        if 0 not in row:
                            self.board = remove_row(self.board, i)
                            cleared_rows += 1
                            floor2.append(i)
                            break
                    else:
                        break
                self.game_clline = cleared_rows
                self.add_cl_lines(cleared_rows, floor2)

                # combo check
                if self.score_flag and not self.block_after_score:
                    self.block_after_score = True
                    self.combo_count += 1
                elif not self.score_flag and self.block_after_score:
                    self.block_after_score = False
                    self.combo_count = 0
                elif self.score_flag and self.block_after_score:
                    self.combo_count += 1

                return True
        self.stone_y = new_y
        return False

    def hard_drop(self, manual):
        """Hard drop logic: drop until collision, then join piece, clear lines, spawn new piece."""
        while 1:
            new_y = self.stone_y
            if not self.gameover:
                new_y = self.stone_y + 1
                cleared_rows = 0
                cur_hole = self.num_hole(self.board)

                if check_collision(self.board, self.stone, (self.stone_x, new_y)):
                    board_top = 0
                    self.hole_score = 0
                    self.minus_score = 0
                    self.plus_score = 0

                    for i in range(rows):
                        for j in range(cols):
                            if self.board[i][j] > 0:
                                board_top = i
                                break
                        if board_top != 0:
                            break
                    if board_top == 0:
                        board_top = rows + 1

                    if new_y > board_top:
                        self.plus_score += 0.05

                    self.board = join_matrixes(self.board, self.stone, (self.stone_x, new_y))

                    if new_y < rows / 2:
                        self.minus_score = (1 - new_y / rows) / 10  # max 0.05

                    if self.chk_block_fit(self.stone, self.stone_x, self.stone_y, self.board):
                        self.plus_score += 0.01
                    else:
                        self.minus_score += 0.01

                    floor1 = 0
                    self.new_stone()
                    while True:
                        for i, row in enumerate(self.board[:-1]):
                            if 0 not in row:
                                self.board = remove_row(self.board, i)
                                cleared_rows += 1
                                floor1 = i
                                break
                            if row.count(0) < 7:
                                global game_score
                                a = row.count(0)
                                game_score -= 0.001 * a
                        else:
                            break
                    self.game_clline = cleared_rows
                    self.add_cl_lines(cleared_rows, floor1)

                    # combo check (same as drop)
                    if self.score_flag and not self.block_after_score:
                        self.block_after_score = True
                        self.combo_count += 1
                    elif not self.score_flag and self.block_after_score:
                        self.block_after_score = False
                        self.combo_count = 0
                    elif self.score_flag and self.block_after_score:
                        self.combo_count += 1

                    return True

            self.stone_y = new_y

        self.stone_y = new_y
        return False

    # ... rest of file continues (step(), reset(), render loop) ...
    def insta_drop(self):
        if not self.gameover and not self.paused:
            while (not self.drop(True)):
                pass

    def rotate_stone(self):
        if not self.gameover and not self.paused:
            new_stone = rotate_clockwise(self.stone)
            if not check_collision(self.board,
                                   new_stone,
                                   (self.stone_x, self.stone_y)):
                self.stone = new_stone

    def n_rotate_stone(self, n):
        self.new_stone_flag = False
        for i in range(n) :
            self.rotate_stone()

    def chk_block_fit(self, stone, x, y, board):
        for m in range(len(stone)):
            for n in range(len(stone[0])):
                if stone[m][n] > 0:
                    if board[y+m+1][x+n] == 0:
                        #print(y+m+1, x+n)
                        #print(board)
                        return False
        return True

    def num_hole(self, board):
        holes = 0
        for n in range(cols):
            for m in range(rows):
                if board[m][n] >= 1:
                    for i in range(rows - m):
                        if board[m + i][n] == 0:
                            holes += 1
                    break

        return holes



    def toggle_pause(self):
        self.paused = not self.paused

    def start_game(self):
        if self.gameover:
            self.init_game()
            self.gameover = False

    # The step is for model training
    def step(self, action):
        current_holes = self.num_hole(self.board)
        self.hole_score = 0
        self.minus_score = 0
        self.plus_score = 0
        post_score = game_score
        self.game_clline = 0
        self.score_flag = False

        '''
        if self.new_stone_flag:
            self.n_rotate_stone(action)
        else :
            if action == 2:
                self.hard_drop(True)
            else :
                self.move_drop(action-2)



        if action==0:
            self.drop(True)
        if action==1:
            self.move(-1)
        if action==2:
            self.move(+1)
        if action==3:
            self.rotate_stone()
        '''
        if action > 20 :
            r_action = action - 21
            self.n_rotate_stone(3)
            self.move_drop(r_action-2)
        elif action > 13 :
            r_action = action - 14
            self.n_rotate_stone(2)
            self.move_drop(r_action - 2)
        elif action > 6 :
            r_action = action - 7
            self.n_rotate_stone(1)
            self.move_drop(r_action - 2)
        else :
            r_action = action
            self.move_drop(r_action - 2)

        self.total_clline += self.game_clline

        self.screen.fill((0, 0, 0))
        self.draw_matrix(self.board, (0, 0))
        self.draw_matrix(self.stone,(self.stone_x, self.stone_y))
        self.draw_matrix(self.next_stone,(cols + 1, 2))

        # Draw screen Boundary
        pygame.draw.line(self.screen, (255, 0, 0), (0, 0), (0, self.height - 1), boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width + boundary_size, 0),
                         (self.width + boundary_size, self.height - 1), boundary_size)
        pygame.draw.line(self.screen, (255, 0, 0), (0, 0), (self.width / 2 + 30, 0), boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width / 2 + 30, 0), (self.width, 0), boundary_size)
        pygame.draw.line(self.screen, (255, 0, 0), (0, self.height - 1), (self.width / 2 + 30, self.height - 1),
                         boundary_size)
        pygame.draw.line(self.screen, (255, 255, 255), (self.width / 2 + 30, self.height - 1),
                         (self.width, self.height - 1),
                         boundary_size)
        pygame.draw.line(self.screen, (255, 0, 0), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                         boundary_size)
        pygame.display.update()

        # all board matrix
        board_screen = copy.deepcopy(self.board)
        stone_m = len(self.stone)
        stone_n = len(self.stone[0])
        for m in range(stone_m):
            for n in range(stone_n):
                if self.stone[m][n] != 0:
                    board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]

        # flatten next stone
        self.next_stone_flat = sum(self.next_stone, [])
        if self.next_stone_flat[0] == 1:
            self.next_stone_flat = self.next_stone_flat + [0, 0]

        # check the board floor is blank or not
        floor = 0
        for k in range(len(board_screen[0])):
            floor += board_screen[rows-1][k]

        self.gameScreen = board_screen

        reward = game_score - post_score

        # reward for board all clear
        if self.allclear_score_flag and reward == 0:
            self.allclear_score_flag = False
            reward += 5
            #print("All Clear!!!")
        if floor == 0 and self.score_flag:
            self.allclear_score_flag = True

        # reward for combo
        if self.combo_score_flag and reward == 0:
            reward += 3
            self.combo_score_flag = False
            #print((self.combo_count-1),"Combo!!!")
        if self.combo_count > 1 and self.score_flag:
            self.combo_score_flag = True

        self.score_flag = False

        # Compose next_state and return consistent (next_state, reward, done)
        # Use the board (current state) as the next_state to match reset()
        next_state = np.array(self.gameScreen, dtype=np.float32)

        # If you want to include the rendered screen instead, replace with
        # next_state = np.array(board_screen, dtype=np.float32)

        #reward adjustments commented out
        #reward -= (self.num_hole(self.board) - current_holes)/20
        return next_state, reward, self.gameover
    def reset(self):
        """Reset the game and return initial state (board)."""
        self.init_game()
        self.gameover = False
        return np.array(self.board, dtype=np.float32)

    def stone_xy(self, n):
        if n==0:
            return self.stone_x
        else:
            return self.stone_y

    # The Run is for only tetris play (not used for training)
    def run(self):
        key_actions = {
            'ESCAPE': self.quit,
            'LEFT': lambda: self.move(-1),
            'RIGHT': lambda: self.move(+1),
            'DOWN': lambda: self.hard_drop(True),
            'UP': self.rotate_stone,
            'p': self.toggle_pause,
            'SPACE': self.start_game,
            'RETURN': self.insta_drop
        }


        self.gameover = False
        self.paused = False

        dont_burn_my_cpu = pygame.time.Clock()

        while 1:
            self.screen.fill((255, 255, 255))
            if self.gameover:
                self.center_msg("""Game Over!\nYour score: %d
Press space to continue""" % game_score)
#                self.center_msg("""Game Over!\nYour score: %d Press space to continue""" % self.score)
            else:
                if self.paused:
                    self.center_msg("Paused")

                else:
                    pygame.draw.line(self.screen,
                                     (255, 255, 255),
                                     (self.rlim + 1, 0),
                                     (self.rlim + 1, self.height - 1))
                    #self.disp_msg("Next:", (self.rlim + cell_size,2))
                    #self.disp_msg("Score: %d\n\nLevel: %d\\nLines: %d" % (self.score, self.level, self.lines),(self.rlim + cell_size, cell_size * 5))
                    self.draw_matrix(self.bground_grid, (0, 0))
                    self.draw_matrix(self.board, (0, 0))
                    self.draw_matrix(self.stone,
                                     (self.stone_x, self.stone_y))
                    self.draw_matrix(self.next_stone,
                                     (cols + 1, 2))

                    # Draw screen Boundary
                    pygame.draw.line(self.screen, (255, 0, 0), (0, 0), (0, self.height - 1), boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (self.width + boundary_size, 0),
                                     (self.width + boundary_size, self.height - 1), boundary_size)
                    pygame.draw.line(self.screen, (255, 0, 0), (0, 0), (self.width/2+30, 0), boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (self.width / 2 +30, 0), (self.width, 0), boundary_size)
                    pygame.draw.line(self.screen, (255, 0, 0), (0, self.height - 1), (self.width / 2 +30, self.height - 1),
                                     boundary_size)
                    pygame.draw.line(self.screen, (255, 255, 255), (self.width / 2 + 30, self.height - 1),(self.width, self.height - 1),
                                     boundary_size)
                    pygame.draw.line(self.screen, (255, 0, 0), (self.rlim + 1, 0), (self.rlim + 1, self.height - 1),
                                     boundary_size)
            if self.render==True:
                pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.USEREVENT + 1:
                    self.drop(False)
                elif event.type == pygame.QUIT:
                    self.quit()
                elif event.type == pygame.KEYDOWN:

                    for key in key_actions:
                        current_score = copy.deepcopy(self.score)
                        if event.key == eval("pygame.K_"
                                             + key):
                            key_actions[key]()
                            #self.gameScreen= pygame.surfarray.array3d(self.screen)

                            # board test
                            ##print(self.board)
                            ##print(self.stone)
                            ##print(self.stone_x)
                            ##print(self.stone_y)
                            board_screen = copy.deepcopy(self.board)
                            stone_m = len(self.stone)
                            stone_n = len(self.stone[0])
                            for m in range(stone_m):
                                for n in range(stone_n):
                                    if self.stone[m][n] != 0:
                                        board_screen[self.stone_y + m][self.stone_x + n] = self.stone[m][n]
                            ##print(board_screen)
                            ##print(numpy.shape(board_screen))

                            # check the board floor is blank or not
                            floor = 0
                            for k in range(len(board_screen[0])):
                                floor += board_screen[rows - 1][k]

                            reward = self.score - current_score

                            # reward for board all clear
                            if self.allclear_score_flag and reward == 0:
                                self.allclear_score_flag = False
                                print("All Clear!!!")
                            if floor == 0 and self.score_flag:
                                self.allclear_score_flag = True

                            # reward for combo
                            if self.combo_score_flag and reward == 0:
                                self.combo_score_flag = False
                                print((self.combo_count - 1), "Combo!!!")
                            if self.combo_count > 1 and self.score_flag:
                                self.combo_score_flag = True

                            self.score_flag = False



            dont_burn_my_cpu.tick(maxfps)
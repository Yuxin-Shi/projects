"""Assignment 2 - Blocky

=== CSC148 Fall 2017 ===
Diane Horton and David Liu
Department of Computer Science,
University of Toronto


=== Module Description ===

This file contains the player class hierarchy.
"""

import random
from typing import Optional, Tuple
import pygame
from renderer import Renderer, BOARD_WIDTH
from block import Block
from goal import Goal

TIME_DELAY = 600


class Player:
    """A player in the Blocky game.

    This is an abstract class. Only child classes should be instantiated.

    === Public Attributes ===
    renderer:
        The object that draws our Blocky board on the screen
        and tracks user interactions with the Blocky board.
    id:
        This player's number.  Used by the renderer to refer to the player,
        for example as "Player 2"
    goal:
        This player's assigned goal for the game.
    """
    renderer: Renderer
    id: int
    goal: Goal

    def __init__(self, renderer: Renderer, player_id: int, goal: Goal) -> None:
        """Initialize this Player.
        """
        self.goal = goal
        self.renderer = renderer
        self.id = player_id

    def make_move(self, board: Block) -> int:
        """Choose a move to make on the given board, and apply it, mutating
        the Board as appropriate.

        Return 0 upon successful completion of a move, and 1 upon a QUIT event.
        """
        raise NotImplementedError


class HumanPlayer(Player):
    """A human player.

    A HumanPlayer can do a limited number of smashes.

    === Public Attributes ===
    num_smashes:
        number of smashes which this HumanPlayer has performed
    === Representation Invariants ===
    num_smashes >= 0
    """
    # === Private Attributes ===
    # _selected_block
    #     The Block that the user has most recently selected for action;
    #     changes upon movement of the cursor and use of arrow keys
    #     to select desired level.
    # _level:
    #     The level of the Block that the user selected
    #
    # == Representation Invariants concerning the private attributes ==
    #     _level >= 0

    # The total number of 'smash' moves a HumanPlayer can make during a game.
    MAX_SMASHES = 1

    num_smashes: int
    _selected_block: Optional[Block]
    _level: int

    def __init__(self, renderer: Renderer, player_id: int, goal: Goal) -> None:
        """Initialize this HumanPlayer with the given <renderer>, <player_id>
        and <goal>.
        """
        super().__init__(renderer, player_id, goal)
        self.num_smashes = 0

        # This HumanPlayer has done no smashes yet.
        # This HumanPlayer has not yet selected a block, so set _level to 0
        # and _selected_block to None.
        self._level = 0
        self._selected_block = None

    def process_event(self, board: Block,
                      event: pygame.event.Event) -> Optional[int]:
        """Process the given pygame <event>.

        Identify the selected block and mark it as highlighted.  Then identify
        what it is that <event> indicates needs to happen to <board>
        and do it.

        Return
           - None if <event> was not a board-changing move (that is, if was
             a change in cursor position, or a change in _level made via
            the arrow keys),
           - 1 if <event> was a successful move, and
           - 0 if <event> was an unsuccessful move (for example in the case of
             trying to smash in an invalid location or when the player is not
             allowed further smashes).
        """
        # Get the new "selected" block from the position of the cursor
        block = board.get_selected_block(pygame.mouse.get_pos(), self._level)

        # Remove the highlighting from the old "_selected_block"
        # before highlighting the new one
        if self._selected_block is not None:
            self._selected_block.highlighted = False
        self._selected_block = block
        self._selected_block.highlighted = True

        # Since get_selected_block may have not returned the block at
        # the requested level (due to the level being too low in the tree),
        # set the _level attribute to reflect the level of the block which
        # was actually returned.
        self._level = block.level

        if event.type == pygame.MOUSEBUTTONDOWN:
            block.rotate(event.button)
            return 1
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                if block.parent is not None:
                    self._level -= 1
                return None

            elif event.key == pygame.K_DOWN:
                if len(block.children) != 0:
                    self._level += 1
                return None

            elif event.key == pygame.K_h:
                block.swap(0)
                return 1

            elif event.key == pygame.K_v:
                block.swap(1)
                return 1

            elif event.key == pygame.K_s:
                if self.num_smashes >= self.MAX_SMASHES:
                    print('Can\'t smash again!')
                    return 0
                if block.smash():
                    self.num_smashes += 1
                    return 1
                else:
                    print('Tried to smash at an invalid depth!')
                    return 0

    def make_move(self, board: Block) -> int:
        """Choose a move to make on the given board, and apply it, mutating
        the Board as appropriate.

        Return 0 upon successful completion of a move, and 1 upon a QUIT event.

        This method will hold focus until a valid move is performed.
        """
        self._level = 0
        self._selected_block = board

        # Remove all previous events from the queue in case the other players
        # have added events to the queue accidentally.
        pygame.event.clear()

        # Keep checking the moves performed by the player until a valid move
        # has been completed. Draw the board on every loop to draw the
        # selected block properly on screen.
        while True:
            self.renderer.draw(board, self.id)
            # loop through all of the events within the event queue
            # (all pending events from the user input)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 1

                result = self.process_event(board, event)
                self.renderer.draw(board, self.id)
                if result is not None and result > 0:
                    # un-highlight the selected block
                    self._selected_block.highlighted = False
                    return 0


class RandomPlayer(Player):
    """
    A RandomPlayer.
    A RandomPlayer can do unlimited numbers of smashes but he chooses every
    move all randomly.
    """
    def __init__(self, renderer: Renderer, player_id: int, goal: Goal) -> None:
        """Initialize this RandomPlayer with the given <renderer>, <player_id>
        and <goal>.
        """
        super().__init__(renderer, player_id, goal)

    def make_move(self, board: Block) -> None:
        """Randomly choose a move to make on the given board,
        and apply it, mutating the Board as appropriate.
        """
        x = random.uniform(0, BOARD_WIDTH)
        y = random.uniform(0, BOARD_WIDTH)
        l = random.randint(0, board.max_depth)
        b = board.get_selected_block((x, y), l)
        b.highlighted = True
        self.renderer.draw(board, self.id)
        pygame.time.wait(TIME_DELAY)

        n = random.randint(1, 3)
        if n == 1:
            b.swap(random.randint(0, 1))
        elif n == 2:
            b.rotate(random.choice([1, 3]))
        else:
            b.smash()
        b.highlighted = False
        self.renderer.draw(board, self.id)


class SmartPlayer(Player):
    """
    A SmartPlayer.
    A SmartPlayer can't smash, but it can smartly compare different moves then
    choose the best to take.
    """

    def __init__(self, renderer: Renderer, player_id: int, goal: Goal,
                 difficulty: int) -> None:
        """Initialize this SmartPlayer with the given <renderer>, <player_id>
        and <goal>.
        """
        super().__init__(renderer, player_id, goal)
        self.difficulty = difficulty

    def make_move(self, board: Block) -> None:
        """Compare several moves and smartly choose a move to
        make on the given board, and apply it,
        mutating the Board as appropriate.
        """
        d = {0: 5, 1: 10, 2: 25, 3: 50, 4: 100, 5: 150}
        if self.difficulty in d:
            t = self.evaluate_moves(board, d[self.difficulty])
        else:
            t = self.evaluate_moves(board, 150)
        t[0].highlighted = True
        self.renderer.draw(board, self.id)
        pygame.time.wait(TIME_DELAY)

        if t[1] == 1:
            t[0].swap(0)
        elif t[1] == 2:
            t[0].swap(1)
        elif t[1] == 3:
            t[0].rotate(1)
        else:
            t[0].rotate(3)
        t[0].highlighted = False
        self.renderer.draw(board, self.id)

    def evaluate_moves(self, board: Block, num_moves: int) -> Tuple[Block, int]:
        """
        Return a tuple to represent the optimal move within the
        next num_moves steps.
        Return 1 for horizontally swap, 2 for vertically swap,
        3 for rotating clockwise, 4 for rotating counterclockwise.

        """
        best = ()
        best_score = 0
        for _ in range(num_moves):
            x = random.uniform(0, BOARD_WIDTH)
            y = random.uniform(0, BOARD_WIDTH)
            l = random.randint(0, board.max_depth)
            b = board.get_selected_block((x, y), l)
            r = random.randint(1, 4)
            if r == 1:
                b.swap(0)
                curr_score = self.goal.score(board)
                if curr_score > best_score:
                    best = (b, r)
                    best_score = curr_score
                b.swap(0)
            elif r == 2:
                b.swap(1)
                curr_score = self.goal.score(board)
                if curr_score > best_score:
                    best = (b, r)
                    best_score = curr_score
                b.swap(1)
            elif r == 3:
                b.rotate(1)
                curr_score = self.goal.score(board)
                if curr_score > best_score:
                    best = (b, r)
                    best_score = curr_score
                b.rotate(3)
            else:
                b.rotate(3)
                curr_score = self.goal.score(board)
                if curr_score > best_score:
                    best = (b, r)
                    best_score = curr_score
                b.rotate(1)
        return best


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'allowed-io': ['process_event'],
        'allowed-import-modules': [
            'doctest', 'python_ta', 'random', 'typing',
            'block', 'goal', 'player', 'renderer',
            'pygame'
        ],
        'max-attributes': 10,
        'generated-members': 'pygame.*'
    })

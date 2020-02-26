"""Assignment 2 - Blocky

=== CSC148 Fall 2017 ===
Diane Horton and David Liu
Department of Computer Science,
University of Toronto


=== Module Description ===

This file contains the Goal class hierarchy.
"""

from typing import List, Tuple
from block import Block


class Goal:
    """A player goal in the game of Blocky.

    This is an abstract class. Only child classes should be instantiated.

    === Attributes ===
    colour:
        The target colour for this goal, that is the colour to which
        this goal applies.
    """
    colour: Tuple[int, int, int]

    def __init__(self, target_colour: Tuple[int, int, int]) -> None:
        """Initialize this goal to have the given target colour.
        """
        self.colour = target_colour

    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given board.

        The score is always greater than or equal to 0.
        """
        raise NotImplementedError

    def description(self) -> str:
        """Return a description of this goal.
        """
        raise NotImplementedError


class BlobGoal(Goal):
    """A goal to create the largest connected blob of this goal's target
    colour, anywhere within the Block.
    """

    def _undiscovered_blob_size(self, pos: Tuple[int, int],
                                board: List[List[Tuple[int, int, int]]],
                                visited: List[List[int]]) -> int:
        """Return the size of the largest connected blob that (a) is of this
        Goal's target colour, (b) includes the cell at <pos>, and (c) involves
        only cells that have never been visited.

        If <pos> is out of bounds for <board>, return 0.

        <board> is the flattened board on which to search for the blob.
        <visited> is a parallel structure that, in each cell, contains:
           -1  if this cell has never been visited
            0  if this cell has been visited and discovered
               not to be of the target colour
            1  if this cell has been visited and discovered
               to be of the target colour

        Update <visited> so that all cells that are visited are marked with
        either 0 or 1.
        """
        if pos[0] >= len(board) or pos[1] >= len(board) or pos[0] < 0 or pos[
                1] < 0:
            return 0
        else:
            count = 0
            if board[pos[0]][pos[1]] == self.colour and \
                            visited[pos[0]][pos[1]] == -1:
                visited[pos[0]][pos[1]] = 1
                c1 = self._undiscovered_blob_size((pos[0] + 1, pos[1]), board,
                                                  visited)
                c2 = self._undiscovered_blob_size((pos[0] - 1, pos[1]), board,
                                                  visited)
                c3 = self._undiscovered_blob_size((pos[0], pos[1] + 1), board,
                                                  visited)
                c4 = self._undiscovered_blob_size((pos[0], pos[1] - 1), board,
                                                  visited)
                count = count + c1 + c2 + c3 + c4 + 1
            elif board[pos[0]][pos[1]] != self.colour:
                visited[pos[0]][pos[1]] = 0
            return count

    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given board.

        The score is always greater than or equal to 0.
        """
        f = board.flatten()
        n = len(f)
        visit = []
        for _ in range(n):
            visit.append([-1] * n)
        size_list = []
        for i in range(n):
            for j in range(n):
                size_list.append(self._undiscovered_blob_size((i, j), f, visit))
        return max(size_list)

    def description(self) -> str:
        """Return a description of this goal.
        """
        return 'BlobGoal.\nThe player must aim for the largest “blob” of a 、' \
               'given colour c.' \
               'A blob is a group of connected blocks with the same colour. ' \
               '\nTwo blocks are connected if their sides touch;' \
               'touching corners doesn’t count. \nThe player’s score is the 、' \
               'number ' \
               'of unit cells in the largest blob of colour c.'


class PerimeterGoal(Goal):
    """A goal to create the most possible units of a given colour c on the \
    outer perimeter of the board.
    """
    def score(self, board: Block) -> int:
        """Return the current score for this goal on the given board.

        The score is always greater than or equal to 0.
        """
        count = 0
        f = board.flatten()
        count += f[0].count(self.colour) + f[-1].count(self.colour)
        for column in f:
            if column[0] == self.colour:
                count = count + 1
            if column[-1] == self.colour:
                count = count + 1
        return count

    def description(self) -> str:
        """Return a description of this goal.
        """
        return 'PerimeterGoal.\nThe player must aim to put the most 、' \
               'possible 、units of a given colour ' \
               'c on the outer perimeter of the board. \nThe player’s 、' \
               'score is the total ' \
               'number of unit cells of colour c that are on the 、' \
               'perimeter. \nThere is ' \
               'a premium on corner cells: they count twice towards the score.'


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        'allowed-import-modules': [
            'doctest', 'python_ta', 'random', 'typing',
            'block', 'goal', 'player', 'renderer'
        ],
        'max-attributes': 15
    })

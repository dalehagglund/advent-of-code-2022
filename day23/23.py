import sys
import typing as ty
from dataclasses import dataclass, field
from enum import Enum
import functools
from functools import partial
import collections
import re
import itertools
from itertools import (
    islice,
    product,
    pairwise,
    zip_longest
)
import abc
from functools import reduce, cmp_to_key
import operator
import collections
from copy import deepcopy
import unittest

def star(f):
    return lambda t: f(*t)

def splitby(predicate, items):
    batch = []
    for item in items:
        if predicate(item):
            yield batch
            batch = []
            continue
        else:
            batch.append(item)
    if batch:
        yield batch
        
def chunked(items, n):
    iters = [ iter(items) ] * n
    return zip(*iters, strict=True)

def sliding_window(items, n):
    window = collections.deque(maxsize=n)
    window.extend(islice(items, n))
    if len(window) == n:
        yield tuple(window)
    for item in items:
        window.append(item)
        yield tuple(window)
    
def trim_newline(line: str) -> str:
    return line[:-1] if line[-1] == '\n' else line
        
def read_sections(file, trim=True) -> list[list[str]]:
    s = iter(file)
    s = map(trim_newline, s)
    s = map(str.rstrip if trim else ident, s)
    s = splitby(lambda line: line == '', s)
    return list(s)
    
def convert_fields(funcs, items: ty.Sequence[ty.Any]):
    return tuple(
        f(item)
        for f, item
        in zip(funcs, items)
        if f is not None
    )

def ident(x): return x
    
def observe(func, items):
    for item in items:
        func(item)
        yield item

def drain(iterable):
    for s in iterable:
        pass

def collect(factory, iterable):
    return factory(iterable)
    
def takeuntil(predicate, items):
    for item in items:
        yield item
        if predicate(item):
            break

def sign(n: int) -> int:
    if   n < 0: return -1
    elif n > 0: return +1
    else:       return 0

def convert_ints(items: list[str]) -> list:
    return [
        int(item) if re.match("^[0-9]*$", item) else item
        for item in items
    ]
    
def product(*ranges, repeat=1):
    ranges = (
        range(r) if isinstance(r, int) else r
        for r in ranges
    )
    return itertools.product(*ranges, repeat=repeat)
    
def nth(n, items): return items[n]
def prod(items): reduce(operator.mul, items, 1)
def mapinner(f, items):
    for item in items:
        yield list(map(f, item))

NORTH = (-1,  0)
SOUTH = (+1,  0)
EAST  = ( 0, +1)
WEST  = ( 0, -1)

EDGE = {
    NORTH: [ (-1, -1), (-1,  0), (-1, +1) ],
    SOUTH: [ (+1, -1), (+1,  0), (+1, +1) ],
    EAST:  [ (-1, +1), ( 0, +1), (+1, +1) ],
    WEST:  [ (-1, -1), ( 0, -1), (+1, -1) ],
}

class Grid:
    def __init__(self, description: list[str], strict=True):
        self._nrows = len(description)
        self._ncols = len(description[0])
        self._grid = [
            list(line)
            for line in description
        ]
        assert all(
            len(row) == self._ncols
            for row in self._grid
        )
        assert all(
            self._grid[i][j] in ".#"
            for i, j in product(self._nrows, self._ncols)
        )
                
        if strict: 
            assert all(
                1 <= i < self._nrows - 1 and 1 <= j < self._ncols - 1
                for i, j in self.elf_positions()
            )
            
    def bounding_box(self): 
        mini = minj = float('+inf')
        maxi = maxj = float('-inf')

        def updatemin(i, j):
            nonlocal mini, minj
            if i < mini: mini = i
            if j < minj: minj = j
        def updatemax(i, j):
            nonlocal maxi, maxj
            if i > maxi: maxi = i
            if j > maxj: maxj = j         
        s = self.elf_positions()
        s = observe(star(updatemin), s)
        s = observe(star(updatemax), s)
        drain(s)
        
        return (mini, minj), (maxi, maxj)
        
    def elf_positions(self):
        return (
            (i, j)
            for i, j in product(self._nrows, self._ncols)
            if self._grid[i][j] == "#"
        )
        #return iter(self._elves)
        
    def at(self, i, j):
        return self._grid[i][j]
        
    def crowded(self, i, j) -> bool:
        assert self._grid[i][j] == "#"
        return any(
            self._grid[i + di][j + dj] == "#"
            for di, dj in product([-1, 0, 1], repeat=2)
            if not (di == dj == 0)
        )
        
    def can_move(self, i, j, dir) -> bool:
        assert self._grid[i][j] == "#"
        return all(
            self._grid[i + di][j + dj] == "."
            for di, dj in EDGE[dir]
        )
        
    def new_position(self, i, j, directions):
        if not self.crowded(i, j): return i, j
        for dir in directions:
            if self.can_move(i, j, dir):
                di, dj = dir
                return i + di, j + dj
        return i, j
    
    def update_elves(self, new_positions):
        for i, j in self.elf_positions():
            self._grid[i][j] = "."
        for i, j in new_positions:
            self._grid[i][j] = "#"
    
    def print(self):
        print(f'nrow = {self._nrows} ncol = {self._ncols}')
        print(f'elves = {list(self.elf_positions())}')
        for row in self._grid:
            print("".join(row))

class GridProposalTests(unittest.TestCase):
    def test_uncrowded(self):
        g = Grid("... .#. ...".split())
        self.assertEqual((1, 1), g.new_position(1, 1, []))
        
    def test_one_direction_free(self):
        directions = [ NORTH, EAST, SOUTH, WEST ]
        cases = [
            [ "... ### .#.", (0, 1) ],
            
            [ ".#. ##. .#.", (1, 2) ],
            
            [ ".#. ### ...", (2, 1) ],
            
            [ ".#. .## .#.", (1, 0) ],
        ]
        for input, expected in cases:
            with self.subTest(input=input, expected=expected):
                g = Grid(input.split(), strict=False)
                self.assertEqual(expected, g.new_position(1, 1, directions))
    
    def test_all_directions_blocked(self):
        directions = [ NORTH, EAST, SOUTH, WEST ]
        g = Grid(".#. ### .#.".split(), strict=False)
        self.assertEqual(
            (1, 1),
            g.new_position(1, 1, directions)
        )

            
class GridQueries(unittest.TestCase):
    def test_crowded_asserts_on_empty_space(self):
        g = Grid("... .#. ...".split())
        self.assertRaises(AssertionError, g.crowded, 0, 0)
    def test_not_crowded(self):
        g = Grid("... .#. ...".split())
        self.assertFalse(g.crowded(1, 1))
    def test_crowded(self):
         g = Grid("..... .##.. ..... ..... .....".split())
         self.assertTrue(g.crowded(1, 1))
         self.assertTrue(g.crowded(1, 2))
    def test_can_move(self):
        g = Grid("... .#. ...".split())
        for direction in [NORTH, SOUTH, EAST, WEST]:
            with self.subTest(dir=direction, edge=EDGE[direction]):
                self.assertTrue(g.can_move(1, 1, direction))
    def test_bounding_box(self):
        cases = [
            [ "... .#. ...", (1, 1), (1, 1) ],
            [ "#.. ... ..#", (0, 0), (2, 2) ],
        ]
        
        for input, exp_ul, exp_lr in cases:
            with self.subTest(input=input, exp_ul=exp_ul, exp_lr=exp_lr):
                g = Grid(input.split(), strict=False)
                ul, lr = g.bounding_box()
                self.assertEqual(exp_ul, ul)
                self.assertEqual(exp_lr, lr)
            
class GridConstructorTests(unittest.TestCase):
    def test_center_elf_3x3_grid(self):
        g = Grid([
                "...",
                ".#.",
                "...",
            ])
        for i, j in product(3, 3):
            expected = "#" if i == j == 1 else "."
            with self.subTest(i=i, j=j, expected=expected):
                if i == j == 1:
                    self.assertEqual(expected, g.at(i, j))
    def test_boundaries_3x3_grid(self):
        for input in [
            "#.. ... ...",
            ".#. ... ...",
            "..# ... ...",
            "... #.. ...",
            "... ..# ...",
            "... ... #..",
            "... ... .#.",
            "... ... ..#",
        ]:
            with self.subTest(input=input):
                self.assertRaises(AssertionError, Grid, input.split())

def compute_proposals(grid, directions):
    proposals = collections.defaultdict(set)
    cur_positions = list(grid.elf_positions())
    for curpos in cur_positions:
        newpos = grid.new_position(*curpos, directions)
        proposals[newpos].add(curpos)        
    assert len(cur_positions) == sum(map(len, proposals.values()))
    assert all(
        sum(map(lambda items: curpos in items, proposals.values())) == 1
        for curpos in cur_positions
    )
    return proposals

def new_positions(proposals):
    new_position = {}
    for new, orig_positions in proposals.items():
        if len(orig_positions) == 1:
            new_position[next(iter(orig_positions))] = new
        else:
            for orig_pos in orig_positions:
                new_position[orig_pos] = orig_pos
    return list(new_position.values())

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    g = Grid(sections[0])
    g.print()
    
    directions = [NORTH, SOUTH, WEST, EAST]
    
    for i in range(3):
        print(f'    iteration {i}: {directions = }')
        proposals = compute_proposals(g, directions)
        g.update_elves(new_positions(proposals))
        directions.insert(-1, directions.pop(0))
        g.print()
    
    
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
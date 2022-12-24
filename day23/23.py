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

class Grid:
    def __init__(self, description: list[str]):
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
        
        self._elves = [
            (i, j)
            for i, j in product(self._nrows, self._ncols)
            if self._grid[i][j] == "#"
        ]
        assert all(
            1 <= i < self._nrows - 1 and 1 <= j < self._ncols - 1
            for i, j in self._elves
        )
        
    def elf_positions(self):
        return iter(self._elves)
        
    def at(self, i, j):
        return self._grid[i][j]
        
    def crowded(self, i, j) -> bool:
        assert self._grid[i][j] == "#"
        return any(
            self._grid[i + di][j + dj] == "#"
            for di, dj in product([-1, 0, 1], repeat=2)
            if not (di == dj == 0)
        )
    
    def print(self):
        print(f'nrow = {self._nrows} ncol = {self._ncols}')
        print(f'elves = {self._elves}')
        for row in self._grid:
            print("".join(row))
            
class GridMovementTests(unittest.TestCase):
    def test_not_crowded(self):
        g = Grid("... .#. ...".split())
        self.assertFalse(g.crowded(1, 1))
    def test_crowded(self):
         g = Grid("..... .##.. ..... ..... .....".split())
         self.assertTrue(g.crowded(1, 1))
         self.assertTrue(g.crowded(1, 2))

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

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    g = Grid(sections[0])
    g.print()
    
    positions = list(g.elf_positions())
    

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
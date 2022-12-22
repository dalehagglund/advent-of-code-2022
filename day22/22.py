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
    
def nth(n, items): return items[n]
def prod(items): reduce(operator.mul, items, 1)
def mapinner(f, items):
    for item in items:
        yield list(map(f, item))
        
RIGHT = ( 0,  +1 )
LEFT  = ( 0,  -1 )
UP    = ( -1,  0 )
DOWN  = ( +1,  0 )

dirvalue = dict(zip([RIGHT, DOWN, LEFT, UP], range(4)))

turn = dict(
    R = {
        RIGHT: DOWN,
        DOWN:  LEFT,
        LEFT:  UP,
        UP:    RIGHT,
    },
    L = {
        RIGHT: UP,
        UP:    LEFT,
        LEFT:  DOWN,
        DOWN:  RIGHT,
    },
)

class Grid:
    def __init__(self, maplines: list[str]):
        self._nrows = len(maplines)
        self._ncols = max(len(line) for line in maplines)
        self._grid = [
            list( line +  " " * (self._ncols - len(line)) )
            for line in maplines
        ]
        assert all(len(row) == self._ncols for row in self._grid)
    
    def advance(self, pos, n, direction) -> tuple[int, int]:
        i, j = self._from_rowcol(*pos)
        assert self._grid[i][j] == "."
        
        print(f'advance: {pos = } {n = } {direction = }')
        
        for _ in range(n):
            #print(f'{(i, j) = }')
            ni, nj = self._step(i, j, direction)
            #print(f'{(ni, nj) = }')
            if self._grid[ni][nj] == "#":
                break
            i, j = ni, nj
        return self._to_rowcol(i, j)
        
    def _step(self, i, j, dir):
        di, dj = dir
        ni, nj = i, j
        while True:
            ni, nj = (ni + di) % self._nrows, (nj + dj) % self._ncols
            if self._grid[ni][nj] != ' ':
                break                
        assert self._grid[ni][nj] != " "
        return ni, nj
    
    def starting_point(self):
        i = 0
        for j in range(self._ncols):
            if self._grid[i][j] == ".":
                return self._to_rowcol(i, j)
        assert False, "no open square in row 0"
        
    def _to_rowcol(self, i, j):
        return (
            self._to_row(i),
            self._to_col(j)
        )
        
    def _from_rowcol(self, row, col):
        return (
            self._from_row(row),
            self._from_col(col)
        )
    
    def _to_row(self, i):
        assert 0 <= i < self._nrows
        return i + 1
    def _to_col(self, j):
        assert 0 <= j < self._ncols
        return j + 1
    def _from_row(self, row):
        assert 1 <= row <= self._nrows
        return row - 1
    def _from_col(self, col):
        assert 1 <= col <= self._ncols
        return col - 1

    def print(self):
        print(f'nrow = {self._nrows} ncol = {self._ncols}')
        for row in self._grid:
            print("".join(row))

def parse_route(route):
    for item in re.split(r'([RL])', route):
        if item == "R" or item == "L":
            yield item
        else:
            yield int(item)
    
def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    maplines = sections[0]
    route = sections[1][0]
    
    grid = Grid(maplines)
    grid.print()
    
    print(turn)
    print(f'starting point = {grid.starting_point()}')
    
    pos = grid.starting_point()
    dir = RIGHT
    
    print(list(parse_route(route)))
    
    for instr in parse_route(route):
        if isinstance(instr, int):
            print(f'advance {instr}')
            pos = grid.advance(pos, instr, dir)
        else:
            print(f'turn {instr} from {dir}')
            dir = turn[instr][dir]
    row, col = pos
    print(f'final {row, col, dir = }')
    
    password = (
        1000 * row +
        4 * col +
        dirvalue[dir]
    )
    print(f'***    {password = }')


def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
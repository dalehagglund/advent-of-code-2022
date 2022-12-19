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
        
@dataclass()
class Interval:
    _lb: int
    _ub: int
    
    def __init__(self, lb: int, ub: int):
        self._lb = lb
        self._ub = ub
    
    def intersects(self, other: 'Interval') -> bool:
        return not (
            other._lb >= self._ub or
            self._lb >= other._ub
        )
    def adjacent(self, other: 'Interval') -> bool:
        return (
            self._ub == other._lb or
            self._lb == other._ub
        )
    
    def merge(self, other: 'Interval') -> 'Interval':
        assert self.intersects(other) or self.adjacent(other)
        return Interval(min(self._lb, other._lb), max(self._ub, other._ub) - 1)
    
    @classmethod
    def empty(cls):
        return Interval(0, 0)
        
    def __len__(self):
        return self._ub - self._lb

class Point(ty.NamedTuple):
    x: int
    y: int

class Tile(abc.ABC):
    def __init__(self, name, short, grid):
        self._name = name
        self._short = short
        self._grid = grid
        self._nrows = len(grid)
        self._ncols = len(grid[0])
    def rock_coords(self):
        for x, y in product(range(self._ncols), range(self._nrows)):
            if self._grid[self._nrows - y - 1][x] == 1:
                yield x, y
    def name(self) -> str: return self._name
    def shape(self) -> list[list[int]]: return self._grid
    def short(self) -> str: return self._short
    def dimensions(self) -> tuple[int, int]: return self._ncols, self._nrows
    def width(self) -> int: return self._ncols
    def height(self) -> int: return self._nrows
    def __repr__(self) -> str:
        return f'Tile(name = {self._name!r}, short = {self._short}, grid = {self._grid!r})'
        
class FlexGrid:
    def __init__(self, seglen):
        self._ymin = 0
        self._seglen = seglen
        self._loseg = self._empty_seg()
        self._hiseg = self._empty_seg()
    def _empty_seg(self):
        return [
            [ "." ] * 7
            for _ in range(self._seglen)
        ]        
    def ymin(self): return self._ymin
    def ymax(self): return self._ymin + 2 * self._seglen
    def at(self, x, y) -> str:
        self._assert_in_bounds(x, y)
        seg, y = self._to_seg(y)
        return seg[y][x]
    def set(self, x, y, value):
        self._assert_in_bounds(x, y)
        seg, y = self._to_seg(y)
        seg[y][x] = value
    def _to_seg(self, y: int):
        if y >= self._seglen: return self._hiseg, y - self._seglen
        else:                 return self._loseg, y
    def _assert_in_bounds(self, x, y):
        if not (0 <= x < 7):
            raise ValueError(f'x out of bounds: {x = }')
        if not (self.ymin() <= y < self.ymax()):
            raise ValueError(f'y out of bounds: {y = }')
            
class FlexGridTests(unittest.TestCase):
    def test_initial_ymin(self):
        g = FlexGrid(4)
        self.assertEqual(0, g.ymin())
    def test_initial_height(self):
        g = FlexGrid(4)
        self.assertEqual(2 * 4, g.ymax() - g.ymin())
    def test_in_bounds_at(self):
        g = FlexGrid(4)
        self.assertEqual(g.at(0, 0), ".")
        self.assertEqual(g.at(0, 7), ".")
    def test_out_of_bounds_indexing(self):
        g = FlexGrid(4)
        cases = [
            (-1, 0),
            (8, 0),
            (0, -1),
            (0, 9),
        ]
        for x, y in cases:
            with self.subTest(x=x, y=y):
                with self.assertRaises(ValueError):
                    g.at(x, y)
    def test_in_bounds_set(self):
        g = FlexGrid(4)
        cases = [
            ("+", 0, 0),
            ("@", 6, 7),
        ]
        for value, x, y in cases:
            with self.subTest(value=value, x=x, y=y):
                g.set(x, y, value)
                self.assertEqual(value, g.at(x, y))
        
class Tower:
    def __init__(self):
        self._grid = [ 
            [ "." ] * 7
            for _ in range(4)
        ]
        self._maxrock = 0
        self._cleartile()
        
    def _cleartile(self):
        self._tile = None
        self._xwidth = self._ywidth = None
        self._tilex = self._tiley = None
        
    def start(self, tile):
        self._tile = tile
        self._xwidth, self._ywidth = tile.width(), tile.height()
        self._tiley = self._maxrock + 3
        self._tilex = min(2, 7 - tile.width())
        self._grid.extend(
            [ "." ] * 7
            for _ in range(max(0, self._tiley + self._tile.height() - len(self._grid)))
        )
        
    def push(self, dir):
        dx = +1 if dir == ">" else -1
        if not self._available(self._tilex + dx, self._tiley): return
        self._tilex += dx
        
    def drop(self) -> bool:
        if not self._available(self._tilex, self._tiley - 1):
            return False
        self._tiley -= 1
        return True
        
    def place(self):
        for dx, dy in self._tile.rock_coords():
            self.set(self._tilex + dx, self._tiley + dy, self._tile.short())
        self._maxrock = max(
            self._maxrock,
            self._tiley + self._tile.height()
        )
        self._cleartile()
        
    def _available(self, x: int, y: int) -> bool:
        #print(f'{x = } {self._tile.width() = }')
        if y < 0: return False
        if x < 0 or x + self._tile.width() > 7: return False
        return all(
            self._grid[y + dy][x + dx] == '.'
            for dx, dy in self._tile.rock_coords()
        )
        
    def at(self, x, y) -> str: return self._grid[y][x]
    def set(self, x, y, char): self._grid[y][x] = char
    
    def print(self, tag=""):
        print(f'*** tower: height {self._maxrock} {tag} ***')
        grid = deepcopy(self._grid)
        if self._tile:
            # print(f'cur tile = {self._tile} pos = {(self._tilex, self._tiley)}')
            for x, y in self._tile.rock_coords():
                grid[self._tiley + y][self._tilex + x] = '@'
          
        for i, row in reversed(list(enumerate(grid))):
            print(f'{"".join(row)}')
    
       
horiz = Tile(
            "horiz", "H",
            [ [ 1, 1, 1, 1 ] ]
        )
cross = Tile(
            "cross", "+",
            [
                [ 0, 1, 0 ],
                [ 1, 1, 1 ],
                [ 0 ,1, 0 ],
            ]
        )
corner = Tile(
            "corner", "\u251b",
            [
                [ 0, 0, 1 ],
                [ 0, 0, 1 ],
                [ 1, 1, 1 ],
            ]
        )
vert = Tile(
            "vert", "V",
            [
                [1], [1], [1], [1],
            ]
        )
square = Tile(
            "square", "Q",
            [
                [ 1, 1 ],
                [ 1, 1 ],
            ]
        )
        
def solve(nrocks: int, jets: str) -> int:
    tower = Tower()
    tiles = itertools.cycle([horiz, cross, corner, vert, square])
    jets = itertools.cycle(jets)
    for i, tile in zip(itertools.count(1), tiles):
        tower.start(tile)
        #tower.print(f'rock {i} {tile.name()} starts falling')
        keep_going = True
        while keep_going:
            dir = next(jets)
            tower.push(dir)
            #tower.print(f'rock {i} push {dir}')
            keep_going = tower.drop()
            if not keep_going: tower.place()
            #\tower.print(f'rock {i} drop')
        #tower.print(f'rock {i} finished falling')
        if i == nrocks:
            break

    return tower._maxrock
        
def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    print(f'   height = {solve(2022, sections[0][0])}')
        

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    print(f'   height = {solve(1000000000000, sections[0][0])}')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
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

class Cave:
    def __init__(self, lines, floor=False):
        def to_coord(s) -> tuple[int, int]:
            x, y = map(int, s.split(","))
            return x, y
        def make_grid(n, m):
            return [
                [ "." for _ in range(m) ]
                for _ in range(n)
            ]
        def draw_segment(start, end):
            sx, sy = start
            ex, ey = end
            if sx == ex:
                for y in range(min(sy, ey), max(sy, ey) + 1):
                    self.set(sx, y, "#") 
            elif sy == ey:
                for x in range(min(sx, ex), max(sx, ex) + 1):
                    self.set(x, sy, "#")
            else:
                assert False, f'huh? bad coords {start}, {end}'
        def process_line(chain):
            s = pairwise(chain)
            s = observe(draw, s)
            drain(s)

        s = iter(lines)
        #s = observe(partial(print, "#1"), s)
        s = map(lambda line: line.split(" -> "), s)
        #s = observe(partial(print, "#2"), s)
        s = mapinner(to_coord, s)
        #s = observe(partial(print, "#3"), s)
        chains = collect(list, s)

        xlow, xhigh = float('+inf'), float('-inf')
        ylow, yhigh = 0, float('-inf')
        def find_bounds(x, y):
            nonlocal xlow, xhigh, ylow, yhigh
            if x < xlow: xlow = x
            elif x > xhigh: xhigh = x
            
            if y < ylow: ylow = y
            elif y > yhigh: yhigh = y

        s = iter(chains)
        s = itertools.chain.from_iterable(s)
        s = observe(star(find_bounds), s)
        s = drain(s)

        assert xlow < xhigh
        assert ylow < yhigh
        assert ylow == 0
        assert xlow <= 500 <= xhigh
        assert ylow <= 0 <= yhigh
                
        xlow -= 1000; xhigh += 1000
        ylow -= 1; yhigh += 1
        
        def make_grid(nrows, ncols):
            return [
                [ "." for _ in range(ncols) ]
                for _ in range(nrows)
            ]

        self._nrows = yhigh - ylow + 1
        self._ncols = xhigh - xlow + 1
        self._xlow, self._xhigh = xlow, xhigh
        self._ylow, self._yhigh = ylow, yhigh
        self._xoffset = -xlow
        self._yoffset = -ylow

        self._grid = make_grid(self._nrows, self._ncols)
        
        self._source = (500, 0)
        
        for chain in chains:
            s = pairwise(chain)
            s = observe(star(draw_segment), s)
            drain(s)
            
        self.set(*self._source, '+')
    
    def source(self): return self._source
    def xlow(self): return self._xlow
    def xhigh(self): return self._xhigh
    def ylow(self): return self._ylow
    def yhigh(self): return self._yhigh
    
    def _widen(self):
        ...
    
    def set(self, x, y, value):
        self._grid[y + self._yoffset][x + self._xoffset] = value
    def at(self, x, y):
        return self._grid[y + self._yoffset][x + self._xoffset]
    def print(self):
        print(
            "\n".join(
                "".join(
                    self.at(x, y)
                    for x in range(self.xlow(), self.xhigh()+1)
                )
                for y in range(self.ylow(), self.yhigh()+1)
            )
        )

    
def nextpos(cave, cur) -> ty.Optional[tuple[int, int]]:
    cx, cy = cur
    moves = [ (0, +1), (-1, +1), (+1, +1) ]
    for dx, dy in moves:
        nx, ny = cx + dx, cy + dy
        if cave.at(nx, ny) == '.': return (nx, ny)
    return None

def drop(cave, floor=False) -> bool:
    sand = cave.source()
    while True:
        sx, sy = sand
        if sy == cave.yhigh():
            if not floor:
                return False
            else:
                cave.set(sx, sy, 'o')
                return True
        newpos = nextpos(cave, sand)
        if newpos is None:
            if (sx, sy) == cave.source():
                print(f'    plugged source')
            cave.set(sx, sy, 'o')
            return True
        sand = newpos

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')

    cave = Cave(sections[0])
    
    grains = 0
    while drop(cave) == True:
        grains += 1
    print(f'    {grains = }')
    
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

    cave = Cave(sections[0], floor=True)
    
    grains = 0
    while cave.at(*cave.source()) == '+':
        drop(cave, floor=True)
        grains += 1
        #print(f'\n\n=== {grains = }')
    print(f'    {grains = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
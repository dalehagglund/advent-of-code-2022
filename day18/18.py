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

class Cell(ty.NamedTuple):
    x: int
    y: int
    z: int
        
directions = [
    (0, 0, 1), ( 0,  0, -1),
    (0, 1, 0), ( 0, -1,  0),
    (1, 0, 0), (-1,  0,  0),
]

def read_cells(lines: list[str]) -> set[Cell]: 
    s = iter(lines)
    s = map(lambda line: line.split(","), s)
    s = map(partial(convert_fields, (int, int, int)), s)
    s = map(star(Cell), s)
    return collect(set, s)

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    cells = read_cells(sections[0])

    area = 0
    for c in cells:
        for dx, dy, dz in directions:
            if (c.x + dx, c.y + dy, c.z + dz) not in cells:
                area += 1
    print(f'    {area = }')

def minmax(items) -> tuple[int, int]:
    min, max = float('+inf'), float('-inf')
    for x in items:
        if x < min: min = x
        if x > max: max = x
    return min, max
    
def flood(minbox, maxbox, droplets) -> set[Cell]:
    xmin, ymin, zmin = minbox
    xmax, ymax, zmax = maxbox
    
    external = set()
    queue = collections.deque()
    
    def neighbours(cell):
        x, y, z = cell
        for dx, dy, dz in directions:
            n = Cell(x + dx, y + dy, z + dz)
            if n in external: continue
            if n in droplets: continue
            if n.x < xmin or n.x > xmax: continue
            if n.y < ymin or n.y > ymax: continue
            if n.z < zmin or n.z > zmax: continue
            yield n
    
    queue.append(Cell(*minbox))
    while queue:
        cell = queue.pop()
        # print('...', cell)
        external.add(cell)
        queue.extend(neighbours(cell))
        
    return external
        
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
        
    droplets = read_cells(sections[0])
    print('read', len(droplets), 'droplets')
    #print(droplets)

    xmin, xmax = minmax(map(partial(nth, 0), droplets))
    ymin, ymax = minmax(map(partial(nth, 1), droplets))
    zmin, zmax = minmax(map(partial(nth, 2), droplets))
    
    xmin -= 1
    ymin -= 1
    zmin -= 1
    
    xmax += 1
    ymax += 1
    zmax += 1
    print('bounding box', (xmin, ymin, zmin), '...', (xmax, ymax, zmax))
    
    external = flood((xmin, ymin, zmin), (xmax, ymax, zmax), droplets)
    print(len(external), 'external nodes')
    #drain(observe(partial(print, "   "), external))

    area = 0
    for d in droplets:
        x, y, z = d
        for dx, dy, dz in directions:
            if (x + dx, y + dy, z + dz) in external:
                area += 1
    print(f'   {area = }')
    
if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
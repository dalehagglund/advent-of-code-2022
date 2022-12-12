import sys
import typing as ty
from dataclasses import dataclass, field
from enum import Enum
import functools
from functools import partial
import collections
import re
import itertools
from itertools import islice, product, pairwise
import abc
from functools import reduce
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
    
class Map:
    def __init__(self, grid, start, end):
        self._heights = grid
        self._nrows = len(grid)
        self._ncols = len(grid[0])
        self._start = start
        self._end = end
        
    def start(self):        return self._start
    def end(self):          return self._end
    def height(self, i, j): return self._heights[i][j]
        
    def neighbours(self, i:  int, j: int):
        for di, dj in [ (-1, 0), (+1, 0), (0, -1), (0, +1) ]:
            ni, nj = i + di, j + dj
            if not(0 <= ni < self._nrows): continue
            if not(0 <= nj < self._ncols): continue
            yield (ni, nj)
            
    def make_distance_grid(self):
        return [
            [ -1 for _ in range(self._ncols) ]
            for _ in range(self._nrows)
        ]
        
    def print(self):
        def letter(i, j):
            if (i, j) == self._start: return "S"
            if (i, j) == self._end: return "E"
            return "abcdefghijklmnopqrstuvwxyz"[self.height(i, j)]
        output = [
            [ letter(i, j) for j in range(self._ncols) ]
            for i in range(self._nrows)
        ]
        print(
            "\n".join(
                " ".join(output_line)
                for output_line in output
            )
        )                
        
    @classmethod
    def parse(cls, lines: list[str]) -> 'Map':
        heights = [
            [ -1 for _ in range(len(lines[0])) ]
            for _ in range(len(lines))
        ]
        start = end = None
        for i, line in enumerate(lines):
            for j, letter in enumerate(line):
                if letter == "S":
                    ht = 0
                    start = (i, j)
                elif letter == "E":
                    ht = 25
                    end = (i, j)
                else:
                    ht = ord(letter) - ord("a")
                heights[i][j] = ht
        assert start is not None and end is not None
        return Map(heights, start, end)
        
def search(map, start, goal, distances, forward=True) -> ty.Optional[int]:
    visited = set()
    queue = collections.deque()
    
    def allowed_move(src, dest):
        return map.height(*dest) <= map.height(*src) + 1
    
    queue.append((0, start))
    while queue:
        # print(f'{queue = }')
        steps, current = queue.popleft()
        # print(f'   {steps = }')
        # print(f'   {current = }')
        # print(f'   {visited = }')
        if current in visited:
            # print(f'   current already visited')
            continue
        i, j = current
        distances[i][j] = steps
        if current == goal:
            return steps
        visited.add(current)

        for neighbour in map.neighbours(*current):
            # print(f'   {neighbour = }')
            if neighbour in visited: continue
            if forward:
                if not allowed_move(current, neighbour): continue
            else:
                if not allowed_move(neighbour, current): continue

            # print(f'      append to {(steps+1, neighbour)}')
            queue.append((steps + 1, neighbour))

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    map = Map.parse(sections[0])
    distances = map.make_distance_grid()

    search(map, map.start(), None, distances)

    print(f'    {distances[map.end()[0]][map.end()[1]] = }')
    
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    
    map = Map.parse(sections[0])
    distances = map.make_distance_grid()

    search(map, map.end(), None, distances, forward=False)
        
    starting_points = [
        (i, j)
        for i, j 
        in product(range(map._nrows), range(map._ncols))
        if map.height(i, j) == 0 and distances[i][j] >= 0
    ]
    best_point = min(starting_points, key=star(lambda i, j: distances[i][j]))

    print(f'    {distances[best_point[0]][best_point[1]] = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
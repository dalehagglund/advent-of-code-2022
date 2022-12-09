import sys
import typing as ty
from dataclasses import dataclass
from enum import Enum
from functools import partial
import collections
import re
from itertools import islice, product, pairwise
import abc
from functools import reduce
import operator

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
            
RIGHT = ( 1,  0)
LEFT  = (-1,  0)
UP    = ( 0,  1)
DOWN  = ( 0, -1)

directions = dict(R=RIGHT, L=LEFT, D=DOWN, U=UP)

@dataclass(frozen=True)
class Knot:
    x: int = 0
    y: int = 0
    
    def move(self, dx: int, dy: int) -> 'Knot':
        assert 0 <= abs(dx) <= 1
        assert 0 <= abs(dy) <= 1
        return Knot(self.x + dx, self.y + dy)
        
    def follow(self, k: 'Knot') -> 'Knot':
        dx, dy = k.x - self.x, k.y - self.y
        dist = max(abs(dx), abs(dy))
        assert 0 <= dist <= 2

        return self.move(sign(dx), sign(dy)) if dist == 2 else self
    
def sign(n: int) -> int:
    if n < 0: return -1
    if n > 0: return 1
    return 0

def read_moves(lines) -> list[tuple[str, int]]:
    s = iter(lines)
    s = map(str.split, s)
    s = map(partial(convert_fields, (str, int)), s)
    return list(s)

def make_chain(n: int) -> list[Knot]:
    return [Knot(0, 0) for _ in range(n)]

def update_chain(chain: list[Knot], direction):
    dx, dy = direction
    chain[0] = chain[0].move(dx, dy)
    for i in range(1, len(chain)):
        chain[i] = chain[i].follow(chain[i-1])

def solve(
        chain: list[Knot],
        moves: list[tuple[str, int]]
) -> int:
    visited = { chain[-1] }
    for dir, nsteps in moves:
        for _ in range(nsteps):
            update_chain(chain, directions[dir])
            visited.add(chain[-1])
    return len(visited)

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    moves = read_moves(sections[0])
    chain = make_chain(2)
    print(f'***    {solve(chain, moves) = }')

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    moves = read_moves(sections[0])
    chain = make_chain(10)
    print(f'***    {solve(chain, moves) = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
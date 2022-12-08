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

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    heights = list(map(list, sections[0]))
    
    s = iter(sections[0])
    s = map(list, s)
    s = map(lambda row: list(map(int, row)), s)
    height = list(s)
    
    nrow = len(height)
    ncol = len(height[0])
    
    def walk_from(i, j, dir):
        di, dj = dir
        i, j = i + di, j + dj
        while 0 <= i < nrow and 0 <= j < ncol:
            yield (i, j)
            i, j = i + di, j + dj
    
    def visible(i: int, j: int) -> bool:
        treeheight  = height[i][j]
        s = iter([ (0, 1), (0, -1), (1, 0), (-1, 0) ])
        s = map(partial(walk_from, i, j), s)
        s = map(list, s)
        vis = any(
            all(treeheight > height[wi][wj] for wi, wj in walk)
            for walk in s
        )
        return vis
        
    def walk_score(treeheight, walk) -> int:
        s = iter(walk)
        s = map(star(lambda i, j: height[i][j]), s)
        s = takeuntil(lambda h: treeheight <= h, s)
        return len(list(s))
        
    def scenic_score(i, j) -> int:
        treeheight  = height[i][j]
        s = iter([ (0, 1), (0, -1), (1, 0), (-1, 0) ])
        s = map(partial(walk_from, i, j), s)
        s = map(list, s)
        s = map(partial(walk_score, treeheight), s)
        score = reduce(operator.mul, s)
        return score
            
    s = product(range(0, nrow), range(0, ncol))
    s = map(star(visible), s)
    s = filter(lambda item: item, s)
    total = len(list(s))
        
    print(f'*** part 1: result = {total} ***')
    
    s = product(range(0, nrow), range(0, ncol))
    s = map(star(scenic_score), s)
    max_score = max(s)
    
    print(f'*** part 2: max scenic score = {max_score}')
 
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
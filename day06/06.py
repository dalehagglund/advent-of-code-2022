import sys
import typing as ty
from dataclasses import dataclass
from enum import Enum
from functools import partial
import re

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
        
def chunk(items, n):
    iters = [ iter(items) ] * n
    return zip(*iters, strict=True)
        
def read_sections(file, trim=True) -> list[list[str]]:
    s = iter(file)
    s = map(
        lambda line: line[:-1] if line[-1] == '\n' else line,
        s
    )
    if trim: s = map(str.rstrip, s)
    s = splitby(lambda line: line == '', s)
    return list(s)
    
def convert_fields(funcs, items: ty.Sequence[ty.Any]):
    return tuple (
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
        
def find_starting_position(s: str, n=4) -> int:
    for i in range(0, len(s)-n):
        start, end = i, i + n
        if len(set(s[start: end])) == n:
            return end
    assert False, "no answer!"
    
def part1(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 1 ***')
    
    for i, line in enumerate(lines):
        offset = find_starting_position(line)
        print(f'{i}: {offset}')

def part2(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 1 ***')
    
    for i, line in enumerate(lines):
        offset = find_starting_position(line, 14)
        print(f'{i}: {offset}')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
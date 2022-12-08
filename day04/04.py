import sys
import typing as ty
from dataclasses import dataclass
from enum import Enum
from functools import partial

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
        
def read_sections(file) -> list[list[str]]:
    s = iter(file)
    s = map(str.rstrip, s)
    s = splitby(lambda line: line == '', s)
    return list(s)
    
def convert_fields(funcs, items: ty.Sequence[ty.Any]):
    return tuple (
        (f(item) if f else item)
        for f, item
        in zip(funcs, items)
    )
    
def to_set(s) -> set[int]:
    low, high = map(int, s.split("-"))
    return set(range(low, high+1))
    
def observe(func, items):
    for item in items:
        func(item)
        yield item

def drain(iterable):
    for s in iterable:
        pass
    
def part1(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 1 ***')
    s = iter(lines)
    s = map(lambda line: line.split(","), s)
    s = map(
            partial(convert_fields, (to_set, to_set)),
            s
    )

    contained = 0
    overlaps = 0
    def count_contains(first, second):
        nonlocal contained
        if first <= second or second <= first:
            contained += 1
    def count_overlaps(first, second):
        nonlocal overlaps
        if first & second:
            overlaps += 1

    s = observe(star(count_contains), s)
    s = observe(star(count_overlaps) , s)
    drain(s)

    print(f'{contained = } {overlaps = }')

def part2(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
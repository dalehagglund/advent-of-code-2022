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
    
priorities = dict(
    zip(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
        range(1, 53)
    )
)

def part1(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 1 ***')

    prio_sum = 0
    for line in lines:
        mid = len(line) // 2
        first, second = set(line[:mid]), set(line[mid:])
        common = (first & second).pop()
        prio_sum += priorities[common]
    print(f'{prio_sum = }')

def part2(fname: str):
    with open(fname) as f:
        lines = read_sections(f)[0]
    print(f'*** part 2 ***')

    prio_sum = 0
    for g in chunk(lines, 3):
        common = (
            set(g[0]) & set(g[1]) & set(g[2])
        ).pop()
        prio_sum += priorities[common]
    print(f'{prio_sum = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
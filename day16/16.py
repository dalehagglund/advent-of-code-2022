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
        
class Valve:
    def __init__(self, name, flow):
        self._name = name
        self._flow = flow
        self._neighbours = set()

    def set_flow(self, flow):
        self._flow = flow
    def add_neighbour(self, v: 'Valve'):
        self._neighbours.add(v)
    
    def name(self): return self._name
    def flow(self): return self._flow
    def neighbours(self): yield from self._neighbours

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
        
    s = iter(sections[0])
    s = map(partial(re.sub, "[=;,]", " "), s)
    s = map(partial(re.sub, "has flow", ""), s)
    s = map(partial(re.sub, "tunnels? leads? to valves?", ""), s)
    s = map(partial(re.sub, "Valve", ""), s)
    s = map(partial(re.sub, "rate", ""), s)
    s = map(str.split, s)
    node_data = collect(list, s)

    valves = dict()
    for vname, flow, *_ in node_data:
        valves[vname] = Valve(vname, int(flow))
    for vname, _, *neighbours in node_data:
        v = valves[vname]
        for n in neighbours:
            v.add_neighbour(valves[n])    

    for v in valves.values():
        print(
            v.name(),
            v.flow(),
            list(map(Valve.name, v.neighbours()))
        )
        
        
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
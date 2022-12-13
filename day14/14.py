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

def in_order(left, right, prefix="") -> int:
    def isint(item): return isinstance(item, int)
    def islist(item): return isinstance(item, list)
    
    #print(f'{prefix}compare {left} vs {right}')
    
    if isint(left) and isint(right):
        if left <  right: return -1
        if left == right: return 0
        if left >  right: return +1
    elif islist(left) and islist(right):
        for l, r in zip_longest(left, right):
            if l is None and r is not None: return -1
            if l is not None and r is None: return +1
            outcome = in_order(l, r, prefix + "  ")
            if outcome != 0:
                return outcome
        return 0
    elif isint(left) and not isint(right):
        return in_order([left], right, prefix + "  ")
    elif not isint(left) and isint(right):
        return in_order(left, [right], prefix + "  ")
    else:
        assert False, f'huh? {left = } {right = }'

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    answer = 0
    for i, section in zip(itertools.count(1), sections): 
        s = iter(section)
        s = map(eval, s)
        left, right = collect(list, s)
       
        if in_order(left, right) == -1:
            answer += i
        
    print(f'    {answer = }')
    
def part2(fname: str):
    with open(fname) as f:
        s = iter(f)
        s = map(trim_newline, s)
        s = filter(lambda line: line != '', s)
        s = map(eval, s)
        packets = collect(list, s)
    print(f'*** part 2 ***')

    dividers = [ [[2]], [[6]] ]
    packets.extend(dividers)
    
    s = iter(packets)
    s = collect(partial(sorted, key=cmp_to_key(in_order)), s)
    s = zip(itertools.count(1), s)
    s = filter(star(lambda _, pkt: pkt in dividers), s)
    s = map(partial(nth, 0), s)
    result = reduce(operator.mul, s)
    
    print(f'    {result = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
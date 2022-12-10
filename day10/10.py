import sys
import typing as ty
from dataclasses import dataclass
from enum import Enum
from functools import partial
import collections
import re
import itertools
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
            
@dataclass
class Regs:
    X:int = 1

def emulate(instructions: list[tuple[str, int]]):
    regs = Regs()
    
    for inst in instructions:
        match inst:
            case ("noop"):
                yield regs.X
            case ("addx", n):
                yield regs.X
                yield regs.X
                regs.X += n
            case _:
                assert False, "huh?"
    
                
def solve(instructions: list[tuple[str, int]]) -> int:
    s = emulate(instructions)
    s = zip(itertools.count(1), s)
    s = filter(star(lambda cyc, _: (cyc - 20) % 40 == 0), s)
    s = map(star(lambda x, y: x * y), s)
    return sum(s)
    return 0
    
def solve2(instructions):
    display = [
        [ "." ] * 40
        for _ in range(6)
    ]

    def draw(row, col, x):
        if col - 1 <= x  <= col + 1:
            display[row][col] = '#'
      
    def row(cycle): return (cycle - 1) // 40
    def col(cycle): return (cycle - 1) %  40

    s = emulate(instructions)
    s = zip(itertools.count(1), s)
    s = map(star(lambda cyc, x: (row(cyc), col(cyc), x)), s)
    s = observe(star(draw), s)
    s = drain(s)
    
    for line in display:
        print("      ", "".join(line))

def decode(line):
    fields = line.split()
    match fields:
        case ["noop"]:
            return ("noop")
        case ["addx", n]:
            return ("addx", int(n))
        case _:
            assert False, "huh?"

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
        
    s = iter(sections[0])
    s = map(decode, s)
    instructions = collect(list, s)
    
    print(f'       result = {solve(instructions)}')

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    s = iter(sections[0])
    s = map(decode, s)
    instructions = collect(list, s)
    
    solve2(instructions)

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
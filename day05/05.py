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
    s = map(str.rstrip if trim else ident, s)
    s = splitby(lambda line: line == '', s)
    return list(s)
    
def convert_fields(funcs, items: ty.Sequence[ty.Any]):
    return tuple (
        f(item)
        for f, item in zip(funcs, items)
        if f is not None
    )

def ident(x): return x
    
def observe(func, items):
    for item in items:
        func(item)
        yield itemro

def drain(iterable):
    for s in iterable:
        pass
    
def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f, trim=False)
    print(f'*** part 1 ***')
    
    drawing = sections[0]
    moves = sections[1]
    
    code = solve_code(drawing, moves, multicrate=False)
    print(f'coded message = {code}')

def solve_code(drawing, moves, *, multicrate=False):     
    nstacks = max(map(int, drawing[-1].split()))
    stacks = [ [] for _ in range(nstacks) ]
    
    pattern = r'''(?:(...) ?)'''
    print(f'{nstacks = }')
    for line in reversed(drawing[:-1]):
        for i, m in enumerate(re.finditer(pattern, line)):
            box = m[1][1]
            if box == ' ': 
                continue
            stacks[i].append(box)
    def printstacks():
        print('stacks:')
        for i, stack in enumerate(stacks):
            print(f'   {i}: {"".join(stack)}') 
    #printstacks()

    s = iter(moves)
    s = map(str.split, s)
    s = map(
            partial(convert_fields, (None, int, None, int, None, int)),
            s
        )
    # s = observe(print, s)
    for n, src, dest in s:
        src -= 1
        dest -= 1
        stacks[dest].extend(
            (ident if multicrate else reversed)((stacks[src][-n:]))
        )
        del stacks[src][-n:]
        #printstacks()
    
    #printstacks()
    code = ''.join(
        stack[-1] if stack else ' '
        for stack in stacks
    )
    return code

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    
    drawing = sections[0]
    moves = sections[1]
    
    code = solve_code(drawing, moves, multicrate=True)
    print(f'coded message = {code}')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
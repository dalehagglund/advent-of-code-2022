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
class Monkey:
    id: int = -1
    items: list[int] = field(default_factory=list)
    update: ty.Callable[[int], int] = lambda x: x
    test: ty.Callable[[int], bool] = lambda x: True
    divisor: int = 0
    adjust: ty.Callable[[int], int] = lambda x: x // 3
    throw_to: dict[bool, int] = field(default_factory=dict)
    inspections: int = 0
    
    def process_items(self):
        #print('Monkey', self.id)
        items, self.items = self.items, []
        for worry in items:
            self.inspections += 1
            #print('   worry', worry) 
            worry = self.update(worry)
            worry = self.adjust(worry)
           
            #print('      updated', worry)
            outcome = self.test(worry)
            #print('      outcome', outcome)
            target = self.throw_to[outcome]
            #print('      throw to', target.id)
            self.throw_to[outcome].receive(worry)
        
    def receive(self, worry):
        self.items.append(worry)
        
def convert_ints(items: list[str]) -> list:
    return [
        int(item) if re.match("^[0-9]*$", item) else item
        for item in items
    ]

def parse_monkey(monkeys: list[Monkey], lines: list[str]):
    def divby(factor):
        return lambda w: w % factor == 0
    def update(action):
        match action:
            case ["*", "old"]: return lambda old: old * old
            case ["*", n]: return lambda old: old * n
            case ["+", "old"]: return lambda old: old + old
            case ["+", n]: return lambda old: old + n
            case _:
                raise ValueError(f'unexpected action: {action}')

    s = iter(lines)
    s = map(partial(re.sub, "[:,]", ""), s)
    s = map(str.split, s)
    s = map(convert_ints, s)
    #s = observe(print, s)  
    
    for fields in s:
        match fields:
            case ["Monkey", ident]:
                monkey = monkeys[ident]
                monkey.id = ident
            case ["Starting", "items", *worries]:
                for w in worries: monkey.receive(w)
            case ["Operation", "new", "=", "old", *action]:
                monkey.update = update(action)
            case ["Test", "divisible", "by", factor]:
                monkey.test = divby(factor)
                monkey.divisor = factor
            case ["If", "true", "throw", "to", "monkey", target]:
                monkey.throw_to[True] = monkeys[target]
            case ["If", "false", "throw", "to", "monkey", target]:
                monkey.throw_to[False] = monkeys[target]
            case _:
                raise ValueError(
                    f'unrecognized line: {fields}'
                )
            
def read_monkeys(
        sections: list[list[str]],
        adjust
) -> list[Monkey]:
    monkeys = [ Monkey(adjust=adjust) for _ in range(len(sections)) ]
    for i, section in enumerate(sections):
        parse_monkey(monkeys, section)
    for m in monkeys:
        print(f'monkey {m.id}: {m.divisor}')
    return monkeys

def update_round(n: int, monkeys: list[Monkey], debug=False):
    for m in monkeys:
        m.process_items()
    if not debug:
        return
    print(f'round {n}')
    for m in monkeys:
        print(f'    monkey {m.id}: {m.items}')
    print(f'   inspections {m.id}: {[m.inspections for m in monkeys]}')

def run(sections, updates, divisor):
    monkeys = read_monkeys(sections, lambda n: n // divisor)
    for i in range(updates):
        update_round(i + 1, monkeys)
    return monkeys

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')

    monkeys = run(sections, 20, 3)
    # for i in range(20):
        # update_round(i + 1, monkeys)
 
    counts = [m.inspections for m in monkeys]
    m1, m2 = sorted(counts)[-2:]
    monkey_business = m1 * m2
    print(f'    {monkey_business = }')

def part2(fname: str):
    import math
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

    def make_adjust(div): return lambda n: n // div

    expected = {
        1: [2, 4, 3, 6 ],
        20: [99, 97, 8, 103 ],
        1000: [ 5204, 4792, 199, 5192 ],
    }       
    modulus = 0
    monkeys = read_monkeys(
        sections, 
        lambda n: int(n % modulus)
    )
    modulus = functools.reduce(operator.mul, (m.divisor for m in monkeys))
    print(f'    {modulus = }')
    
    matches = set()
    for i in range(1, 10001):
        update_round(i, monkeys, debug=False)
        
        #if i not in expected.keys(): continue
        
        #observed = [m.inspections for m in monkeys]
        # print(f'{divisor = } {observed = }')
        #if observed != expected[i]: break
        #matches.add(i)
    if len(matches) > 1:
        print(f'{matches = }')
            
        
    counts = [m.inspections for m in monkeys]
    print(counts)
    m1, m2 = sorted(counts)[-2:]
    monkey_business = m1 * m2
    print(f'    {monkey_business = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
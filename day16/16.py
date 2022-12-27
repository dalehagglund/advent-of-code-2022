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
import unittest

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

    def add_neighbour(self, v: 'Valve') -> None:
        self._neighbours.add(v)
    
    def name(self) -> str: return self._name
    def flow(self) -> int: return self._flow
    def neighbours(self): yield from self._neighbours
    
    def __str__(self):
        return f'{self._name}/{self._flow}'
    def __repr__(self):
        return str(self)

class ValveTests(unittest.TestCase):
    def test_basic_construction(self):
        v = Valve('v', 15)
        self.assertEqual('v', v.name())
        self.assertEqual(15, v.flow())
        self.assertEqual(0, len(list(v.neighbours())))
    def test_add_neighbours(self):
        v = Valve('v', 15)
        n1 = Valve('n1', 10)
        n2 = Valve('n2', 20)
        v.add_neighbour(n1)
        v.add_neighbour(n2)
        
        neighbours = set(v.neighbours())
        self.assertEqual(2, len(neighbours))
        self.assertTrue(n1 in neighbours)
        self.assertTrue(n2 in neighbours)
    
@dataclass(frozen=True)        
class State:
    ttl: int
    loc: Valve
    open: set[Valve] = field(default_factory=set)
    released_so_far: int = 0
    
    def next_states(self):
        newttl = self.ttl - 1
        released = self.released_so_far + self.flow_rate()
        if self.loc not in self.open and self.loc.flow() > 0:
            yield State(
                newttl, self.loc, self.open | {self.loc}, released
            )
        for v in self.loc.neighbours():
            yield State(
                newttl, v, self.open, released
            )
    
    def flow_rate(self):
        return sum(v.flow() for v in self.open)

    def __hash__(self):
        return hash((
            self.ttl,
            self.loc,
            self.released_so_far,
            *self.open
        ))
        
def make_valves(descr: list[tuple[str, int]]) -> list[Valve]:
    return [ Valve(name, flow) for name, flow in descr ]

class TestStateFlow(unittest.TestCase):
    def test_default_flow_so_far_is_zero(self):
        v = Valve('v', 10)
        s = State( ttl=1, loc=v, open=set() )
        self.assertEqual(0, s.released_so_far)

    def test_flow_rate(self):
        v, w = make_valves(zip("vw", (0, 10)))
        s = State(ttl=1, loc=v, open={w})
        self.assertEqual(10, s.flow_rate())
        
    def test_next_states_increase_total_release(self):
        v, w, x = make_valves(zip("vwx", (0, 10, 20)))
        v.add_neighbour(w)
        
        s = State(ttl=1, loc=v, open={x})
        for ns in s.next_states():
            self.assertEqual(20, ns.released_so_far)
        
class NextStateTests(unittest.TestCase):
    def test_doesnt_turn_on_valve_if_flow_is_zero(self):
        v = Valve('v', 0)
        state = State(0, loc=v, open=set())
        successors = set(state.next_states())
        self.assertEqual(0, len(successors))
        
    def test_turns_on_valve_if_flow_isnt_zero(self):
        v = Valve('v', 10)
        state = State(1, loc=v, open=set())
        
        successors = set(state.next_states())
        self.assertEqual(1, len(successors))
        
        next_state = next(iter(successors))
        self.assertEqual(0, next_state.ttl)
        self.assertTrue(next_state.loc in next_state.open)
        self.assertEqual(1, len(next_state.open))

    @unittest.skip("defer")
    def test_open_doesnt_shrink(self):
        self.assertFalse(True)

    def test_follows_neighbours(self):
        v, w, x = make_valves(zip("vwx", (0, 20, 30)))
        v.add_neighbour(w)
        v.add_neighbour(x)
        
        s = State(1, loc=v, open=set())
        successors = set(s.next_states())
        
        self.assertEqual(2, len(successors))
        self.assertEqual(
            {"w", "x"},
            { ns.loc.name() for ns in successors }
        )
        self.assertEqual(
            {w, x},
            {ns.loc for ns in successors}
        )
        self.assertTrue(all(ns.ttl == 0 for ns in successors))
        
def search(valves, start, ttl, verbose=False):
    import builtins
    queue: list[State] = []
    best_release = float('-inf')
    max_flow_rate = sum(v.flow() for v in valves.values())
    
    def estimated_release(s: State) -> int:
        return (
            s.released_so_far +
            s.flow_rate() * s.ttl
        )
        
    def best_possible_release(s: State) -> int:
        return (
            s.released_so_far +
            max_flow_rate * s.ttl
        )
    
    print = builtins.print if verbose else lambda *_: 1
    print(f'search: {start = !s} {ttl = }')
    
    counter = itertools.count(1)
    
    
    queue.append( State(ttl, start, set()) )
    while queue:
        n = next(counter)
        
        if n % 1000 == 0:
            builtins.print(f'{n}: {len(queue) = } {best_release = }')
        if verbose:
            print('queue = [')
            for item in queue:
                print(f'   {item}')
            print(']')
            
        state = queue.pop(0)
        print(f'examining {state = }')
        
        if state.ttl == 0:
            print(f'   ttl == 0 {best_release = }')
            released = state.released_so_far
            if  released > best_release:
                best_release = released
            continue
        elif best_possible_release(state) < best_release:
            print(f"pruning: can't reach current best_release")
        else:
            print(f'   expanding')
            for succ in state.next_states():
                print(f'      {succ = }')
                queue.append( succ )
            # sorting is O(n log n) so this loop could be O(n^2) ???
            queue.sort(key=estimated_release, reverse=True)
        
    print(f'no more states: {best_release = }')
        
def read_valves(lines: list[str]) -> dict[str, Valve]:
    s = iter(lines)
    s = map(partial(re.sub, "[=;,]", " "), s)
    s = map(partial(re.sub, "has flow", ""), s)
    s = map(partial(re.sub, "tunnels? leads? to valves?", ""), s)
    s = map(partial(re.sub, "Valve", ""), s)
    s = map(partial(re.sub, "rate", ""), s)
    s = map(str.split, s)
    node_data = collect(list, s)

    valves = {
        vname: Valve(vname, int(flow))
        for vname, flow, *_
        in node_data
    }
    for vname, _, *neighbours in node_data:
        v = valves[vname]
        for n in neighbours:
            v.add_neighbour(valves[n])    

    return valves

def print_valves(valves: dict[str, Valve]):
    for v in valves.values():
        print(f'node: {v.name()} flow: {v.flow()}')
        print(f'   neighbours: {" ".join(n.name() for n in v.neighbours())}')

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')    

    valves = read_valves(sections[0])

    print_valves(valves)
    start = valves['AA']
    
    search(valves, start, 30, verbose=False)
        
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
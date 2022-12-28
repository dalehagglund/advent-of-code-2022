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
import functools
from functools import reduce, cmp_to_key
import operator
import collections
import unittest
import heapq
import time

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
    non_progress: set[Valve] = field(default_factory=set)
    
    @functools.cache
    def flow_rate(self):
        return sum(v.flow() for v in self.open)

    def __hash__(self):
        return hash((
            self.ttl,
            self.loc,
            self.released_so_far,
            *self.open,
            *self.non_progress,
        ))
        
def next_states(state, max_possible_flow):
    make_state = partial(
        State, 
        ttl = state.ttl - 1,
        released_so_far = state.released_so_far + state.flow_rate()
    )
    
    if state.flow_rate() == max_possible_flow:
        # all valves are on, just sit here
        yield make_state(
            loc=state.loc,
            open=state.open,
            non_progress=state.non_progress
        )
        return
    
    if state.loc not in state.open and state.loc.flow() > 0:
        # the total flow rate only increases here, so reset the nonprogress set
        yield make_state(
            loc = state.loc, 
            open = state.open | {state.loc},
            non_progress = set()
        )

    for v in state.loc.neighbours():
        # only expand v if it doesn't form a no-progress cycle.
        # note that none of the states generated here increase the
        # flow rate
        if v in state.non_progress: continue
        yield make_state(
            loc = v,
            open = state.open,
            non_progress = state.non_progress | {state.loc}
        )
    
class NextStateTests(unittest.TestCase):
    def test_doesnt_turn_on_valve_if_flow_is_zero(self):
        v = Valve('v', 0)
        state = State(0, loc=v, open=set())
        successors = set(next_states(state, float('+inf')))
        self.assertEqual(0, len(successors))
        
    def test_turns_on_valve_if_flow_isnt_zero(self):
        v = Valve('v', 10)
        state = State(1, loc=v, open=set())
        
        successors = set(next_states(state, float('+inf')))
        self.assertEqual(1, len(successors))
        
        next_state = next(iter(successors))
        self.assertEqual(0, next_state.ttl)
        self.assertTrue(next_state.loc in next_state.open)
        self.assertEqual(1, len(next_state.open))

    @unittest.skip("defer")
    def test_open_doesnt_shrink(self):
        self.assertFalse(True)
        
    def test_stay_put_if_max_flow_achieved(self):
        v, w, x = make_valves(zip("vwx", (0, 20, 30)))
        v.add_neighbour(w)
        v.add_neighbour(x)
        
        s = State(ttl=1, loc=v, open={w, x})
        successors = set(next_states(s, 20 + 30))
        
        self.assertEqual(1, len(successors))
        ns = next(iter(successors))
        
        self.assertEqual(s.ttl - 1, ns.ttl)
        self.assertEqual(s.loc, ns.loc)
        self.assertEqual(s.open, ns.open)

    def test_follows_neighbours(self):
        v, w, x = make_valves(zip("vwx", (0, 20, 30)))
        v.add_neighbour(w)
        v.add_neighbour(x)
        
        s = State(1, loc=v, open=set())
        successors = set(next_states(s, float('+inf')))
        
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
        
    def test_detect_cycle_with_no_progress(self):
        v, w = make_valves(zip("vw", (0, 0)))
        v.add_neighbour(w)
        w.add_neighbour(v)
        
        s1 = State(10, loc=v, open=set())
        s2 = exactly(1, next_states(s1, float('+inf')))
        
        self.assertEqual(0, len(set(next_states(s2, float('+inf')))))
 
def exactly(n, items):
    pairs = zip(range(n), items, strict=True)
    if n == 1:
        return next(pairs)[1]
    
    return list(next(pairs)[1] for _ in range(n))

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
        for ns in next_states(s, float('+inf')):
            self.assertEqual(20, ns.released_so_far)

class PrioQueue:
    def __init__(self, key=None):
        self._key = key if key is not None else lambda x: x
        self._pq = []
        self._counter = itertools.count()
    def insert(self, item):
        heapq.heappush(self._pq, self._to_entry(item))
    def pop(self):
        return self._to_item(heapq.heappop(self._pq))
    def _to_entry(self, item):
        return (self._key(item), next(self._counter), item)
    def _to_item(self, entry):
        return entry[-1]
    def __len__(self):
        return len(self._pq)
    
class PrioQueueTests(unittest.TestCase):
    def test_constructor(self):
        pq = PrioQueue()
    def test_len_empty_qp_is_zero(self):
        pq = PrioQueue()
        self.assertEqual(0, len(pq))
    def test_len_is_one_after_insert(self):
        pq = PrioQueue()
        pq.insert(3)
        self.assertEqual(1, len(pq))
    def test_pop_returns_last_item(self):
        pq = PrioQueue()
        pq.insert(1)
        self.assertEqual(1, pq.pop())
    def test_min_out_first(self):
        pq = PrioQueue()
        pq.insert(3)
        pq.insert(2)
        pq.insert(1)
        self.assertEqual(1, pq.pop())
    def test_use_keyfunc_to_invert_heap_order(self):
        pq = PrioQueue(key=lambda x: -x)
        pq.insert(3)
        pq.insert(2)
        pq.insert(1)
        self.assertEqual(3, pq.pop())

def search(valves, start, ttl, verbose=False):
    import builtins
    
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
    
    def priority_heuristic(s: State):
        return -estimated_release(s)
    
    queue: PrioQueue = PrioQueue(key=priority_heuristic)
    
    counter = itertools.count(1)
    total_pruned = pruned = 0
    total_appended = appended = 1
    
    elapsed_start = lap_start = time.perf_counter()
    
    queue.insert( State(ttl, start, set()) )
    while queue:
        n = next(counter)
        
        if n % 1000 == 0:
            t = time.perf_counter()
            total_appended += appended
            total_pruned += pruned
            
            builtins.print(
                f'elapsed {t - elapsed_start:6.3f}',
                f'lap {t  - lap_start:5.3f}',
                f'{n}:',
                'qlen', len(queue),
                'pruned', pruned,
                'appended', appended,
                'best', best_release
            )
            appended = pruned = 0
            lap_start = time.perf_counter()
        if verbose:
            print('queue = [')
            for item in queue:
                print(f'   {item}')
            print(']')
            
        state = queue.pop()
        if verbose: print(f'examining {state = }')
        
        if state.ttl == 0:
            if verbose: print(f'   ttl == 0 {best_release = }')
            released = state.released_so_far
            if  released > best_release:
                best_release = released
        elif best_possible_release(state) < best_release:
            pruned += 1
            if verbose: print(f"   pruning: can't reach current best_release")
        else:
            print(f'   expanding')
            for succ in next_states(state, max_flow_rate):
                if verbose: print(f'      {succ = }')
                queue.insert( succ )
                appended += 1            
        
    builtins.print(f'no more states: {best_release = } {total_appended = } {total_pruned = }    ')
        
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
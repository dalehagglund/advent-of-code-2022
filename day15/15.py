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

def sign(n: int) -> int:
    if   n < 0: return -1
    elif n > 0: return +1
    else:       return 0

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
        
@dataclass()
class Interval:
    _lb: int
    _ub: int
    
    def __init__(self, lb: int, ub: int):
        self._lb = lb
        self._ub = ub
    def intersects(self, other: 'Interval') -> bool:
        if other._ub <= self._lb: return False
        if self._ub <= other._lb: return False
        return True
        
    def adjacent(self, other: 'Interval') -> bool:
        return (
            self._ub == other._lb or
            self._lb == other._ub
        )
        
    def _contains(self, other) -> bool:
        return self._lb <= other._lb <= other._ub <= self._ub
        
    def intersection(self, other: 'Interval') -> 'Interval':
        if not self.intersects(other): return Interval.empty()
        
        if self._contains(other):
            return other
        elif other._contains(self):
            return self
        elif self._lb <= other._lb <= self._ub <= other._ub:
            return Interval(other._lb, self._ub)
        elif other._lb <= self._lb <= other._ub <= self._ub:
            return Interval(self._lb, other._ub)
        
        assert False
        
    def contains(self, x: int) -> bool:
        return self._lb <= x < self._ub
    
    def merge(self, other: 'Interval') -> 'Interval':
        assert self.intersects(other) or self.adjacent(other)
        return Interval(min(self._lb, other._lb), max(self._ub, other._ub))
    
    @classmethod
    def empty(cls):
        return Interval(0, 0)
        
    def __len__(self):
        return self._ub - self._lb
        
class IntervalTests(unittest.TestCase):
    def test_intersects(self):
        cases = [
            ( True, Interval(0, 17), Interval(11, 21) ),
        ]
        for expected, intv1, intv2 in cases:
            with self.subTest(expected=expected, intv1=intv1, intv2=intv2):
                self.assertEqual(expected, intv1.intersects(intv2))
                self.assertEqual(expected, intv2.intersects(intv1))

    def test_adjacent(self):
        i1 = Interval(1, 2)
        i2 = Interval(2, 3)
        self.assertTrue(i1.adjacent(i2))
        self.assertTrue(i2.adjacent(i1))
    def test_not_adjacent(self):
        i1 = Interval(1, 2)
        i2 = Interval(3, 4)
        self.assertFalse(i1.adjacent(i2))
        self.assertFalse(i2.adjacent(i1))
    def test_empty_contains_nothing(self):
        empty = Interval(3, 3)
        self.assertFalse(empty.contains(3))
        self.assertFalse(empty.contains(2))
        self.assertFalse(empty.contains(4))
    def test_empty_contains(self):
        intv = Interval(0, 10)
        for i in range(10):
            self.assertTrue(intv.contains(i))
        self.assertFalse(intv.contains(-1))
        self.assertFalse(intv.contains(10))
    def test_merge_adjacent(self):
        Intv = Interval
        self.assertEqual(
            Intv(0, 10),
            Intv(0, 5).merge(Intv(5, 10)))
        self.assertEqual(
            Intv(0, 10),
            Intv(5, 10).merge(Intv(0, 5)))
    def test_intersection(self):
        cases = [
            # interval1        interval2         expected
            ( Interval(0, 5),  Interval.empty(), Interval.empty() ),
            ( Interval(0, 5),  Interval(10, 15), Interval.empty() ),
            ( Interval(0, 5),  Interval(0, 5),   Interval(0, 5)   ),
            ( Interval(0, 5),  Interval(2, 3),   Interval(2, 3)   ),
            ( Interval(0, 5),  Interval(0, 4),   Interval(0, 4)   ),
            ( Interval(0, 5),  Interval(1, 5),   Interval(1, 5)   ),
            ( Interval(0, 5),  Interval(4, 9),   Interval(4, 5),  ),
            
            ( Interval(0, 21), Interval(-3, 8),  Interval(0, 8)   )
        ]
        for intv1, intv2, expected in cases:
            with self.subTest(intv1=intv1, intv2=intv2, expected=expected):
                self.assertEqual(expected, intv1.intersection(intv2))
                self.assertEqual(expected, intv2.intersection(intv1))
    def test_merge_overlapped(self):
        cases = [
            (Interval(0, 7), Interval(6, 10)),
            (Interval(0, 7), Interval(5, 10)),
            (Interval(0, 7), Interval(1, 10)),
        ]
        
        for case, (iv1, iv2) in enumerate(cases):
            with self.subTest(case):
                self.assertEqual(Interval(0, 10), iv1.merge(iv2))
                self.assertEqual(Interval(0, 10), iv2.merge(iv1))
    def test_containing(self):
        cases = [
            (Interval(0, 7), Interval(0, 1)),
            (Interval(0, 7), Interval(1, 2)),
            (Interval(0, 7), Interval(1, 7)),
            (Interval(0, 7), Interval(6, 7)),
            (Interval(0, 7), Interval(7, 7)),
            (Interval(0, 7), Interval(0, 7)),
        ]
        for case, (iv1, iv2) in enumerate(cases):
            with self.subTest(case):
                self.assertEqual(iv1, iv1.merge(iv2))
                self.assertEqual(iv1, iv2.merge(iv1))

class Point(ty.NamedTuple):
    x: int
    y: int
    
    def dist(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)
        
@dataclass
class Sensor:
    loc: Point
    beacon: Point
    _radius: int = field(init=False)
    
    def __post_init__(self):
        self._radius = self.loc.dist(self.beacon)
    
    def can_see(self, p: Point) -> bool:
        return self._radius >= self.loc.dist(p)
        
    def radius(self): return self._radius
        
    def xinterval_at(self, y: int) -> Interval:
        ydelta = abs(y - self.loc.y)
        if ydelta > self._radius: return Interval.empty()
        xdelta = abs(self._radius - ydelta)
        assert xdelta + ydelta == self._radius
        # print('...', self.loc, f'{y = }', xdelta, ydelta)
        return Interval(self.loc.x - xdelta, self.loc.x + xdelta + 1)

class SensorTests(unittest.TestCase):
    def test_radius_zero(self):
        sensor = Sensor(Point(0, 0), Point(0, 0))
        self.assertEqual(0, sensor.radius())
    def test_xinterval_with_radius_zero(self):
        sensor = Sensor(Point(0, 0), Point(0, 0))
        interval = sensor.xinterval_at(0)
        self.assertEqual(Interval(0, 1), interval)
        self.assertEqual(Interval.empty(), sensor.xinterval_at(1))
    def test_xintervals_radius_one(self):
        sensor = Sensor(Point(0, 0), Point(1, 0))
        self.assertEqual(1, sensor.radius())
        cases = [
            ( 2, Interval.empty()),
            ( 1, Interval(0, 1)),
            ( 0, Interval(-1, 2)),
            (-1, Interval(0, 1)),
            (-2, Interval.empty()),
        ]
        for y, expected in cases:
            with self.subTest(y=y, expected=expected):
                self.assertEqual(expected, sensor.xinterval_at(y))
    def test_xintervals_radius_zero(self):
        sensor = Sensor(Point(0, 0), Point(0, 0))
        self.assertEqual(0, sensor.radius())
        cases = [
            ( 1, Interval.empty()),
            ( 0, Interval(0, 1)),
            (-1, Interval.empty()),
        ]
        for y, expected in cases:
            with self.subTest(y=y, expected=expected):
                self.assertEqual(expected, sensor.xinterval_at(y))
        
def read_sensors(lines: list[str]) -> list[Sensor]:
    s = iter(lines)
    s = map(partial(re.sub, "[=,:]", " "), s)
    s = map(str.split, s)
    s = map(lambda t: (nth(3, t), nth(5, t), nth(11, t), nth(13, t)), s)
    s = map(partial(convert_fields, (int, int, int, int)), s)
    s = map(lambda t: Sensor(Point(t[0], t[1]), Point(t[2], t[3])), s)
    return collect(list, s)
    
def merge_intervals(intervals: list[Interval]) -> list[Interval]:
    if len(intervals) == 0: return []
    intervals = sorted(intervals, key=lambda intv: intv._lb)
    merged = [intervals[0]]
    for item in intervals[1:]:
        cur = merged[-1]
        # print(f'    {cur = } {item = }')
        if cur.intersects(item) or cur.adjacent(item):
            merged[-1] = cur.merge(item)
        else:
            merged.append(item)
    return merged

def part1(fname: str, ytarget: int):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    sensors = read_sensors(sections[0])
    
    s = map(lambda intv: intv.xinterval_at(ytarget), sensors)
    s = filter(lambda intv: len(intv) > 0, s)
    intervals = collect(list, s)
        
    intervals.sort(key=lambda s: s._lb)
    drain(observe(print, intervals))
    
    print()
    
    merged = [intervals[0]]
    for item in intervals[1:]:
        cur = merged[-1]
        print(f'{cur = } {item = }')
        if cur.intersects(item) or cur.adjacent(item):
            merged[-1] = cur.merge(item)
        else:
            merged.append(item)
    drain(observe(print, merged))
    
    sensor_points = { s.loc for s in sensors }
    beacon_points = { s.beacon for s in sensors }
    excluded = sum(len(intv) for intv in merged)
    
    print(f'initial {excluded = }')
    for p in sensor_points | beacon_points:
        #print("considering point", p)
        if p.y != ytarget: continue
        if any(intv.contains(p.y) for intv in intervals):
            print('   hit', p)
            excluded -= 1
            
    print(f'    {excluded = }')

def find_intervals(sensors, xrange: Interval, y: int) -> list[Interval]:
    for s in sensors:
        intv = s.xinterval_at(y)
        intv = intv.intersection(xrange)
        if intv == Interval.empty(): continue
        yield intv

def part2(fname: str, boxsize: int):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')
    
    sensors = read_sensors(sections[0])
    xinterval = Interval(0, boxsize + 1)
    
    print(f'{xinterval = }')
    point = None
    for y in range(boxsize + 1):
        if y % 10000 == 0: print(f'{y = }')
        intervals = list(find_intervals(sensors, xinterval, y))
        intervals = merge_intervals(intervals)
        #print(f'    {intervals = }')
        assert 0 <= len(intervals) <= 2
        if len(intervals) == 0:
            continue
        elif len(intervals) == 2:
            point = Point(x=intervals[0]._ub, y=y)
            break
        elif intervals[0] == xinterval:
            continue
        elif intervals[0]._lb == 0:
            assert len(intervals[0]) == boxsize - 1
            point = Point(x=intervals[0]._ub, y=y)
            break
        elif intervals[0]._lb == 1:
            assert len(intervals[0]) == boxsize - 1
            point = Point(x=0, y=y)
            break
        else:
            assert False

    print(f'***   found empty point {point}')
    x, y = point
    print(f'***   tuning freq = {x * 4000000 + y}')

if __name__ == '__main__':
    part1(sys.argv[1], int(sys.argv[2]))
    part2(sys.argv[1], int(sys.argv[3]))
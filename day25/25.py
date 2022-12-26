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
from copy import deepcopy
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
    
def product(*ranges, repeat=1):
    ranges = (
        range(r) if isinstance(r, int) else r
        for r in ranges
    )
    return itertools.product(*ranges, repeat=repeat)
    
def nth(n, items): return items[n]
def prod(items): reduce(operator.mul, items, 1)
def mapinner(f, items):
    for item in items:
        yield list(map(f, item))


def snafu_to_int(digits: str) -> int:
    snafu_values = dict(zip("012-=", [0, 1, 2, -1, -2]))
    
    value = 0
    for place, digit in zip(itertools.count(0), digits[::-1]):
        value += snafu_values[digit] * (5 ** place)
        
    return value

def int_to_snafu(n: int) -> str:
    
    snafu_digit = dict(zip([0, 1, 2, 3, 4], "012=-"))
    snafu_adjustment = dict(zip([0, 1, 2, 3, 4], [0, 0, 0, 1, 1]))
    
    digits = []
    q = n
    while True:
        q, r = divmod(q, 5)
        q += snafu_adjustment[r]
        digits.append(snafu_digit[r])
        if q == 0:
            break

    return "".join(reversed(digits))
    
class IntToSnafuTests(unittest.TestCase):
    def test_int_to_snafu(self):
        cases = [
            ( 0, "0" ),
            ( 1, "1" ),
            ( 2, "2" ),
            ( 3, "1=" ),
            ( 4, "1-" ),
            ( 5, "10" ),
            ( 6, "11" ),
            ( 7, "12" ),
            ( 8, "2=" ),
            ( 9, "2-" ),
            ( 10, "20" ),
            ( 314159265, "1121-1110-1=0" ),
        ]
        for n, expected in cases:
            with self.subTest(n=n, expected=expected):
                self.assertEqual(
                    expected,
                    int_to_snafu(n)
                )
    
class SnafuToIntTests(unittest.TestCase):
    def test_single_digits(self):
        cases = [
            ( "2", 2 ),
            ( "1", 1 ),
            ( "0", 0 ),
            ( "-", -1 ),
            ( "=", -2 ),
        ]
        for digit, expected in cases:
            with self.subTest(digit=digit, expected=expected):
                self.assertEqual(expected, snafu_to_int(digit))
                
    def test_multi_digits(self):
        cases = [
            ( "1=", 3 ),
            ( "1-", 4 ),
            ( "1121-1110-1=0", 314159265 ),
        ]
        for digits, expected in cases:
            with self.subTest(digits=digits, expected=expected):
                self.assertEqual(expected, snafu_to_int(digits))

def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    s = iter(sections[0])
    s = map(snafu_to_int, s)
    total = sum(s)
    
    print(f'    sum = {total}')
    print(f'    snafu = {int_to_snafu(total)}')

def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
    
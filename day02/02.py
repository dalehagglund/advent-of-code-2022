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
        
def read_sections(file) -> list[list[str]]:
    s = iter(file)
    s = map(str.rstrip, s)
    s = splitby(lambda line: line == '', s)
    return list(s)
    
def convert_fields(funcs, items: ty.Sequence[ty.Any]):
    return tuple (
        (f(item) if f else item)
        for f, item in
        zip(funcs, items)
    )

class Choice(Enum):
    Rock = 0
    Paper = 1
    Scissors = 2

defeats = {
    Choice.Rock: Choice.Scissors,
    Choice.Scissors: Choice.Paper,
    Choice.Paper: Choice.Rock
}

shape_score = {
    Choice.Rock: 1,
    Choice.Paper: 2,
    Choice.Scissors: 3,
}
            
decode = dict(
    A=Choice.Rock, B=Choice.Paper, C=Choice.Scissors,
    X=Choice.Rock, Y=Choice.Paper, Z=Choice.Scissors,
)

def to_choice(code: str) -> Choice:
    return decode[code]

def score(other: Choice, me: Choice) -> int:
    if other == me:
        return 3 + shape_score[me]
    elif defeats[me] == other:
        return 6 + shape_score[me]
    else:
        return 0 + shape_score[me]

def response(other, outcome) -> Choice:
    if outcome == "X":
        return defeats[other]
    elif outcome == "Y":
        return other
    elif outcome == "Z":
        for winner, loser in defeats.items():
            if loser == other:
                return winner
    assert False, f"never get here {other = } {outcome  = }"
    
def part1(fname):
    with open(fname) as f:
        sections = read_sections(f)
    assert len(sections) == 1
    print('*** part1 ***')
    s = sections[0]
    s = map(str.split, s)
    s = map(partial(convert_fields, (to_choice, to_choice)), s)
    strategy = list(s)

    total_score = 0
    for other, me in strategy:
        total_score += score(other, me)
    print(f'{total_score = }')
        

def part2(fname):
    with open(fname) as f:
        sections = read_sections(f)
    print('*** part2 ***')
    s = sections[0]
    s = map(str.split, s)
    s = map(partial(convert_fields,(to_choice, None)), s)
    info = list(s)
    
    total_score = 0
    for other, outcome in info:
        me = response(other, outcome)
        total_score += score(other, me)
    print(f'{total_score = }')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
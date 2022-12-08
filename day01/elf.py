import sys

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

def reader(file):
    s = iter(file)
    s = map(str.rstrip, s)
    s = splitby(lambda line: line == '', s)
    elves = list(s)
    for i in range(len(elves)):
        elves[i] = list(map(int, elves[i]))
    return elves
    
def part1(fname):
    with open(fname) as f:
        elves = reader(f)
    s = elves
    s = map(sum, s)
    maxcal = max(s)
    
    print(f'*** part 1: max elf is carrying {maxcal} calories')
    
def part2(fname):
    with open(fname) as f:
        elves = reader(f)
    s = elves
    s = map(sum, s)
    max3cal = sum(sorted(s)[-3:])
    
    print(f'*** part 2: max 3 elves are carrying {max3cal} calories')

def main():
    part1(sys.argv[1])
    part2(sys.argv[1])

if __name__ == '__main__':
    main()
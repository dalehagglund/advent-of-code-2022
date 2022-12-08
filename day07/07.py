import sys
import typing as ty
from dataclasses import dataclass
from enum import Enum
from functools import partial
import collections
import re
from itertools import islice
import abc

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
        window.append(x)
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

    
class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str: ...
    @abc.abstractmethod
    def size(self) -> int: ...
    
class File(Node):
    def __init__(self, name, size):
        self._name = name
        self._size = size
    def name(self): return self._name
    def size(self): return self._size
    
    def __repr__(self):
        return f'File({self.name()}, {self.size()})'
    
class Dir(Node):
    def __init__(self, name):
        self._name = name
        self._children: dict[str, Node] = {}
        self._changed = False
        self._size = 0
    def name(self) -> str: return self._name
    def add_child(self, child: Node):
        self._changed = True
        self._children[child.name()] = child
    def size(self):
        if not self._changed:
            return self._size
        self._size = sum(child.size() for child in self._children.values())
        self._changed = False
        return self._size
    def lookup_dir(self, name: str) -> Node:
        return self._children[name]
    def walk_dirs(self):
        yield self
        for child in self._children.values():
            if isinstance(child, Dir):
                yield from child.walk_dirs()
        
    def lstree(self, depth=0):
        prefix = '  ' * depth if depth > 0 else ''
        print(
            prefix,
            '-',
            self.name(),
            f'(dir, size={self.size()}'
        )
        for node in self._children.values():
            if isinstance(node, Dir):
                node.lstree(depth = depth+1)
                continue
            print(
                prefix + '  ',
                node.name(),
                f'(file, size={node.size()}'
            )
            
        
    def __repr__(self):
        return f'Dir({self.name()}, {[k for k in self._children]})'
        
def parse_filetree(lines: list[str]) -> Dir:
    s = iter(lines)
    s = map(str.split, s)    
    commands = collect(list, s)
    
    stack = []
    root = None
    
    def push(item): stack.append(item)
    def pop(): return stack.pop()
    def top(): return stack[-1]
    
    for cmd in commands:
        #print('Cmd:', cmd)

        match cmd:
            case ('$', 'cd', '..'):
                pop()
            case ('$', 'cd', '/'):
                root = Dir('/')
                push(root)
            case ('$', 'cd', name):
                push(top().lookup_dir(name))
            case ('$', 'ls'):
                pass
            case ('dir', name):
                top().add_child(Dir(name))
            case (size, name) if re.match('^[0-9]*$', size):
                top().add_child(File(name, int(size)))
            case _:
                assert False, f'unexpected input {cmd}'

        #print('Stack:')
        #for item in reversed(stack):
        #    print(f'    {item}')
    
    #root.lstree()
    
    return root
    
            
def part1(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 1 ***')
    
    root = parse_filetree(sections[0])
    
    s = root.walk_dirs()
    s = filter(lambda d: d.size() <= 100000, s)
    
    print(f'*** result = {sum(d.size() for d in s)}')
 
def part2(fname: str):
    with open(fname) as f:
        sections = read_sections(f)
    print(f'*** part 2 ***')

    root = parse_filetree(sections[0])    

    disk_space = 70000000
    min_required = 30000000
    
    unused = disk_space - root.size()
    extra_needed = min_required - unused
    
    s = root.walk_dirs()
    s = filter(lambda d: d.size() >= extra_needed, s)
    target = sorted(s, key=lambda d: d.size())[0]
    print(f'*** result = {target.size()}')

if __name__ == '__main__':
    part1(sys.argv[1])
    part2(sys.argv[1])
# task_tracker.py
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Tuple, Dict, Iterable, Callable, Any, Optional as TypingOptional
from functools import reduce, lru_cache
import pytest

# -------------------------
# Domain entities (immutable)
# -------------------------
@dataclass(frozen=True)
class Task:
    id: int
    title: str
    status: str         # e.g., "todo", "in_progress", "done"
    priority: int       # higher = more important (1..5)
    tags: Tuple[str, ...]
    subtasks: Tuple[int, ...]  # ids of subtasks
    estimate_minutes: int      # estimated effort

# -------------------------
# Simple dataset (immutable tuples / mapping)
# -------------------------
SAMPLE_TASKS: Tuple[Task, ...] = (
    Task(1, "Implement login", "todo", 4, ("backend", "auth"), (2, 3), 120),
    Task(2, "Design login UI", "done", 3, ("frontend", "ui"), (), 60),
    Task(3, "Add OAuth", "in_progress", 5, ("backend","oauth"), (4,), 90),
    Task(4, "Register with Google", "todo", 2, ("oauth",), (), 45),
    Task(5, "Write tests", "todo", 3, ("tests",), (), 80),
)

def tasks_to_dict(tasks: Iterable[Task]) -> Dict[int, Task]:
    """Pure function: convert iterable of tasks to dict mapping id->Task."""
    return {t.id: t for t in tasks}

# -------------------------
# Pure transformation functions
# -------------------------
def add_tag(task: Task, tag: str) -> Task:
    """Return new Task with tag added (idempotent if tag already present)."""
    if tag in task.tags:
        return task
    return replace(task, tags=tuple(task.tags + (tag,)))

def change_status(task: Task, new_status: str) -> Task:
    """Return new Task with updated status."""
    return replace(task, status=new_status)

def bump_priority(task: Task, delta: int = 1) -> Task:
    """Return new Task with adjusted priority, clipped to [1, 5]."""
    new_p = max(1, min(5, task.priority + delta))
    return replace(task, priority=new_p)

# -------------------------
# Map / Filter / Reduce pipelines (pure)
# -------------------------
def filter_tasks(tasks: Iterable[Task], predicate: Callable[[Task], bool]) -> Tuple[Task, ...]:
    return tuple(filter(predicate, tasks))

def map_tasks(tasks: Iterable[Task], mapper: Callable[[Task], Task]) -> Tuple[Task, ...]:
    return tuple(map(mapper, tasks))

def reduce_tasks(tasks: Iterable[Task], reducer: Callable[[Any, Task], Any], initial: Any) -> Any:
    return reduce(reducer, tasks, initial)

# Examples of useful pipelines
def total_estimate(tasks: Iterable[Task]) -> int:
    return reduce_tasks(tasks, lambda acc, t: acc + t.estimate_minutes, 0)

def tasks_with_tag(tasks: Iterable[Task], tag: str) -> Tuple[Task, ...]:
    return filter_tasks(tasks, lambda t: tag in t.tags)

# -------------------------
# Lambdas and closures
# -------------------------
def make_status_filter(status: str) -> Callable[[Task], bool]:
    """Closure that returns a predicate checking status equality."""
    return lambda t: t.status == status

def make_priority_at_least(min_priority: int) -> Callable[[Task], bool]:
    return lambda t: t.priority >= min_priority

# -------------------------
# Recursion: aggregate over subtasks
# -------------------------
def make_tasks_lookup(tasks: Iterable[Task]) -> Dict[int, Task]:
    return tasks_to_dict(tasks)

def validate_no_cycles(tasks_dict: Dict[int, Task]) -> bool:
    """Simple cycle detection for the subtasks graph (returns False if cycle)."""
    visited = set()
    rec_stack = set()

    def dfs(node_id: int) -> bool:
        if node_id not in tasks_dict:
            return True  # unknown node considered not a cycle here
        if node_id in rec_stack:
            return False
        if node_id in visited:
            return True
        visited.add(node_id)
        rec_stack.add(node_id)
        for sub in tasks_dict[node_id].subtasks:
            if not dfs(sub):
                return False
        rec_stack.remove(node_id)
        return True

    for tid in list(tasks_dict.keys()):
        if tid not in visited:
            if not dfs(tid):
                return False
    return True

def total_estimate_recursive(task_id: int, tasks_dict: Dict[int, Task]) -> int:
    """Recursively sum estimates of a task and its subtasks (no memo)."""
    if task_id not in tasks_dict:
        return 0
    t = tasks_dict[task_id]
    return t.estimate_minutes + sum(total_estimate_recursive(sid, tasks_dict) for sid in t.subtasks)

# -------------------------
# Memoized version (Lab 3)
# -------------------------
def make_memoized_total_estimate(tasks_dict: Dict[int, Task]):
    """Return a memoized function bound to this tasks_dict using lru_cache."""
    @lru_cache(maxsize=None)
    def total(task_id: int) -> int:
        if task_id not in tasks_dict:
            return 0
        t = tasks_dict[task_id]
        return t.estimate_minutes + sum(total(sid) for sid in t.subtasks)
    return total

# -------------------------
# Option / Result containers (Lab 4)
# -------------------------
class Option:
    """Simple Maybe container: Option(value) or None_"""
    def __init__(self, value: TypingOptional[Any]):
        self._v = value

    @staticmethod
    def some(value: Any) -> 'Option':
        return Option(value)

    @staticmethod
    def none() -> 'Option':
        return Option(None)

    def is_some(self) -> bool:
        return self._v is not None

    def map(self, func: Callable[[Any], Any]) -> 'Option':
        if self.is_some():
            return Option.some(func(self._v))
        return Option.none()

    def and_then(self, func: Callable[[Any], 'Option']) -> 'Option':
        if self.is_some():
            return func(self._v)
        return Option.none()

    def unwrap_or(self, default: Any) -> Any:
        return self._v if self.is_some() else default

    def __repr__(self):
        return f"Option({self._v!r})"

class Result:
    """Simple Either-like container: Ok(value) or Err(error)"""
    def __init__(self, ok: bool, value: Any):
        self.ok = ok
        self.value = value

    @staticmethod
    def Ok(value: Any) -> 'Result':
        return Result(True, value)

    @staticmethod
    def Err(error: Any) -> 'Result':
        return Result(False, error)

    def map(self, func: Callable[[Any], Any]) -> 'Result':
        if self.ok:
            try:
                return Result.Ok(func(self.value))
            except Exception as e:
                return Result.Err(e)
        return self

    def and_then(self, func: Callable[[Any], 'Result']) -> 'Result':
        if self.ok:
            try:
                return func(self.value)
            except Exception as e:
                return Result.Err(e)
        return self

    def unwrap_or(self, default: Any) -> Any:
        return self.value if self.ok else default

    def __repr__(self):
        typ = "Ok" if self.ok else "Err"
        return f"Result.{typ}({self.value!r})"

# Utility using Option/Result
def find_task(tasks_dict: Dict[int, Task], task_id: int) -> Option:
    return Option.some(tasks_dict[task_id]) if task_id in tasks_dict else Option.none()

def try_find_task(tasks_dict: Dict[int, Task], task_id: int) -> Result:
    return Result.Ok(tasks_dict[task_id]) if task_id in tasks_dict else Result.Err(f"Task {task_id} not found")

# -------------------------
# Composition helpers
# -------------------------
def compose(*funcs: Callable):
    """Right-to-left composition: compose(f,g,h)(x) => f(g(h(x)))"""
    def composed(x):
        return reduce(lambda acc, f: f(acc), reversed(funcs), x)
    return composed

# -------------------------
# Demo usage
# -------------------------
def demo():
    print("=== Demo Task Tracker (Labs 1-4 combined) ===")
    tasks = SAMPLE_TASKS
    tasks_dict = make_tasks_lookup(tasks)
    print("All tasks:", tasks)
    print("Total raw estimate (all tasks):", total_estimate(tasks))
    print("Tasks with 'backend' tag:", tasks_with_tag(tasks, "backend"))
    high_priority = filter_tasks(tasks, make_priority_at_least(4))
    print("High priority tasks:", high_priority)

    # recursion and memoization
    print("Total estimate for task 1 (recursive):", total_estimate_recursive(1, tasks_dict))
    memo_total = make_memoized_total_estimate(tasks_dict)
    print("Total estimate for task 1 (memoized):", memo_total(1))

    # Option / composition example
    opt = find_task(tasks_dict, 3).map(lambda t: t.title.upper())
    print("Option mapped title for task 3:", opt)

    # Result example
    r = try_find_task(tasks_dict, 99).map(lambda t: t.title)
    print("Try find missing task 99:", r)

# -------------------------
# Tests (pytest)
# -------------------------
def test_add_tag_idempotent():
    t = Task(10, "A", "todo", 1, ("x",), (), 10)
    t2 = add_tag(t, "y")
    t3 = add_tag(t2, "y")
    assert "y" in t2.tags
    assert t2 == t3  # adding same tag again doesn't change

def test_change_status_returns_new():
    t = Task(11, "B", "todo", 2, (), (), 5)
    t2 = change_status(t, "in_progress")
    assert t.status == "todo"
    assert t2.status == "in_progress"
    assert t != t2

def test_pipeline_total_estimate_and_filters():
    tasks = SAMPLE_TASKS
    backend = tasks_with_tag(tasks, "backend")
    assert all("backend" in t.tags for t in backend)
    assert total_estimate(backend) == 120 + 90  # tasks 1 and 3

def test_closure_filters():
    tasks = SAMPLE_TASKS
    f = make_status_filter("todo")
    todos = filter_tasks(tasks, f)
    assert all(t.status == "todo" for t in todos)

def test_recursive_total_and_memoization():
    tasks_dict = make_tasks_lookup(SAMPLE_TASKS)
    raw = total_estimate_recursive(1, tasks_dict)
    memo = make_memoized_total_estimate(tasks_dict)
    memo_val = memo(1)
    assert raw == memo_val
    # validate memoization speeds up repeated calls (functional: same answer)
    assert memo(1) == memo_val  # second call hits cache

def test_validate_no_cycles_detects_cycle():
    # create simple cycle
    a = Task(100, "A", "todo", 1, (), (101,), 10)
    b = Task(101, "B", "todo", 1, (), (100,), 5)
    d = make_tasks_lookup((a, b))
    assert not validate_no_cycles(d)

def test_option_result_behavior():
    tasks_dict = make_tasks_lookup(SAMPLE_TASKS)
    some = find_task(tasks_dict, 2).map(lambda t: t.title)
    assert some.unwrap_or("X") == "Design login UI"
    none = find_task(tasks_dict, 999)
    assert none.unwrap_or("missing") == "missing"
    r_ok = try_find_task(tasks_dict, 2)
    assert r_ok.ok
    r_err = try_find_task(tasks_dict, 999)
    assert not r_err.ok

# -------------------------
# Run demo if executed directly
# -------------------------
if __name__ == "__main__":
    demo()
    print("\nRun 'pytest -q task_tracker.py' to run tests.")

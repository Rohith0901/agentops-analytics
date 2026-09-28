"""
tasks.py — the benchmark's task bank.

Fifteen small, self-contained coding tasks that mirror CodeShift's real task_type
taxonomy (see ARCHITECTURE.md): code_migration, bug_fix, test_generation, refactor,
feature_build. Each task is a plain dict (not a class) so it is trivial to add more
or export to JSON.

Fields
------
id              : short stable identifier, e.g. "t01"
task_type       : one of code_migration | bug_fix | test_generation | refactor | feature_build
prompt          : the natural-language instruction sent to the LLM (this *is* what
                  gets sent to Ollama as the user prompt)
entry_func      : the name of the top-level Python function the model must define
tests           : hidden unit tests — a list of short, assert-based Python snippets.
                  Each snippet is exec'd independently against the model's code so we
                  can report partial credit (some asserts pass, others fail).
reference_solution : a correct implementation. Used only by run_benchmark.py --mock
                  to synthesize plausible "correct" model outputs — never shown to a
                  real model and never used to judge a real model's answer.
buggy_solution  : a deliberately-wrong implementation with a plausible, subtle bug.
                  Used only by --mock to synthesize plausible "buggy" model outputs.

Grading is always done by executing the model's *own* code against `tests`; the
reference/buggy solutions never participate in real (non-mock) grading.
"""

from __future__ import annotations

TASKS: list[dict] = [
    # ------------------------------------------------------------------ code_migration
    {
        "id": "t01",
        "task_type": "code_migration",
        "entry_func": "average",
        "prompt": (
            "Convert this Python 2 function to idiomatic Python 3. Fix the print "
            "statement, make division return a float, and remove any Python-2-only "
            "syntax. Keep the function name `average`.\n\n"
            "```python2\n"
            "def average(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    print \"total:\", total\n"
            "    return total / len(nums)\n"
            "```\n"
            "Return only the Python 3 function, no print statements."
        ),
        "tests": [
            "assert average([1, 2, 3]) == 2.0",
            "assert average([2, 2, 2, 2]) == 2.0",
            "assert abs(average([1, 2]) - 1.5) < 1e-9",
        ],
        "reference_solution": (
            "def average(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    return total / len(nums)\n"
        ),
        "buggy_solution": (
            "def average(nums):\n"
            "    total = 0\n"
            "    for n in nums:\n"
            "        total += n\n"
            "    return total // len(nums)\n"  # integer division bug reintroduced
        ),
    },
    {
        "id": "t02",
        "task_type": "code_migration",
        "entry_func": "filter_even_squares",
        "prompt": (
            "Translate this Java 8 stream pipeline into an idiomatic Python function "
            "named `filter_even_squares(nums)` that returns a sorted (ascending) list "
            "of the squares of the even numbers in `nums`.\n\n"
            "```java\n"
            "List<Integer> result = nums.stream()\n"
            "    .filter(n -> n % 2 == 0)\n"
            "    .map(n -> n * n)\n"
            "    .sorted()\n"
            "    .collect(Collectors.toList());\n"
            "```"
        ),
        "tests": [
            "assert filter_even_squares([1, 2, 3, 4]) == [4, 16]",
            "assert filter_even_squares([]) == []",
            "assert filter_even_squares([5, 7]) == []",
            "assert filter_even_squares([-2, 3, 4]) == [4, 16]",
        ],
        "reference_solution": (
            "def filter_even_squares(nums):\n"
            "    return sorted(n * n for n in nums if n % 2 == 0)\n"
        ),
        "buggy_solution": (
            "def filter_even_squares(nums):\n"
            "    # bug: filters odd numbers instead of even\n"
            "    return sorted(n * n for n in nums if n % 2 != 0)\n"
        ),
    },
    {
        "id": "t03",
        "task_type": "code_migration",
        "entry_func": "credit_charge",
        "prompt": (
            "Modernize this legacy COBOL-style balance-deduction routine into a "
            "Python function `credit_charge(balance, amount)` that returns the new "
            "balance after deducting `amount`. If `amount` exceeds `balance`, raise "
            "a `ValueError` instead of allowing an overdraft.\n\n"
            "```\n"
            "PROCEDURE CHARGE-ACCOUNT.\n"
            "    IF AMOUNT > BALANCE\n"
            "        DISPLAY 'INSUFFICIENT FUNDS'\n"
            "    ELSE\n"
            "        SUBTRACT AMOUNT FROM BALANCE.\n"
            "```"
        ),
        "tests": [
            "assert credit_charge(100, 40) == 60",
            "assert credit_charge(50, 50) == 0",
            (
                "try:\n"
                "    credit_charge(10, 20)\n"
                "    assert False, 'expected ValueError'\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
        "reference_solution": (
            "def credit_charge(balance, amount):\n"
            "    if amount > balance:\n"
            "        raise ValueError('insufficient funds')\n"
            "    return balance - amount\n"
        ),
        "buggy_solution": (
            "def credit_charge(balance, amount):\n"
            "    # bug: never raises, allows negative balance\n"
            "    return balance - amount\n"
        ),
    },
    {
        "id": "t04",
        "task_type": "code_migration",
        "entry_func": "top_n_scores",
        "prompt": (
            "Translate this .NET LINQ expression into a Python function "
            "`top_n_scores(scores, n)` that returns the `n` highest scores, "
            "sorted descending.\n\n"
            "```csharp\n"
            "var top = scores.OrderByDescending(s => s).Take(n).ToList();\n"
            "```"
        ),
        "tests": [
            "assert top_n_scores([5, 1, 9, 3], 2) == [9, 5]",
            "assert top_n_scores([1, 2, 3], 0) == []",
            "assert top_n_scores([4], 5) == [4]",
        ],
        "reference_solution": (
            "def top_n_scores(scores, n):\n"
            "    return sorted(scores, reverse=True)[:n]\n"
        ),
        "buggy_solution": (
            "def top_n_scores(scores, n):\n"
            "    # bug: ascending order instead of descending\n"
            "    return sorted(scores)[:n]\n"
        ),
    },
    # ------------------------------------------------------------------------ bug_fix
    {
        "id": "t05",
        "task_type": "bug_fix",
        "entry_func": "sum_to_n",
        "prompt": (
            "This function is supposed to return the sum of all integers from 1 to "
            "`n` inclusive, but it has an off-by-one bug. Fix it. Keep the name "
            "`sum_to_n`.\n\n"
            "```python\n"
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
            "```"
        ),
        "tests": [
            "assert sum_to_n(5) == 15",
            "assert sum_to_n(1) == 1",
            "assert sum_to_n(0) == 0",
        ],
        "reference_solution": "def sum_to_n(n):\n    return sum(range(1, n + 1))\n",
        "buggy_solution": (
            "def sum_to_n(n):\n"
            "    total = 0\n"
            "    for i in range(1, n):\n"
            "        total += i\n"
            "    return total\n"
        ),
    },
    {
        "id": "t06",
        "task_type": "bug_fix",
        "entry_func": "is_palindrome",
        "prompt": (
            "This palindrome checker fails on strings with spaces, punctuation, or "
            "mixed case (e.g. 'A man a plan a canal Panama'). Fix `is_palindrome(s)` "
            "so it ignores case and non-alphanumeric characters.\n\n"
            "```python\n"
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
            "```"
        ),
        "tests": [
            "assert is_palindrome('Race car') is True",
            "assert is_palindrome('hello') is False",
            "assert is_palindrome('A man a plan a canal Panama') is True",
        ],
        "reference_solution": (
            "def is_palindrome(s):\n"
            "    cleaned = ''.join(ch.lower() for ch in s if ch.isalnum())\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        "buggy_solution": "def is_palindrome(s):\n    return s == s[::-1]\n",
    },
    {
        "id": "t07",
        "task_type": "bug_fix",
        "entry_func": "safe_divide",
        "prompt": (
            "This function crashes with ZeroDivisionError instead of handling the "
            "zero-denominator case gracefully. Fix `safe_divide(a, b)` so it returns "
            "`None` when `b == 0`, otherwise returns `a / b`.\n\n"
            "```python\n"
            "def safe_divide(a, b):\n"
            "    return a / b\n"
            "```"
        ),
        "tests": [
            "assert safe_divide(10, 2) == 5.0",
            "assert safe_divide(5, 0) is None",
            "assert safe_divide(-9, 3) == -3.0",
        ],
        "reference_solution": (
            "def safe_divide(a, b):\n"
            "    return None if b == 0 else a / b\n"
        ),
        "buggy_solution": "def safe_divide(a, b):\n    return a / b\n",
    },
    # ------------------------------------------------------------------ test_generation
    {
        "id": "t08",
        "task_type": "test_generation",
        "entry_func": "test_addition_properties",
        "prompt": (
            "Write a function named `test_addition_properties` that takes no "
            "arguments. Using plain `assert` statements, verify: (1) addition is "
            "commutative for at least 3 different pairs of numbers, and (2) adding "
            "0 to a number returns that same number, for at least 3 different "
            "numbers. Catch any `AssertionError` internally: return `True` if every "
            "assertion passed, `False` if any failed."
        ),
        "tests": [
            "assert test_addition_properties() is True",
            "assert callable(test_addition_properties)",
        ],
        "reference_solution": (
            "def test_addition_properties():\n"
            "    try:\n"
            "        assert 2 + 3 == 3 + 2\n"
            "        assert 5 + 7 == 7 + 5\n"
            "        assert -1 + 4 == 4 + (-1)\n"
            "        assert 10 + 0 == 10\n"
            "        assert 0 + 0 == 0\n"
            "        assert -5 + 0 == -5\n"
            "        return True\n"
            "    except AssertionError:\n"
            "        return False\n"
        ),
        "buggy_solution": (
            "def test_addition_properties():\n"
            "    # bug: does not catch AssertionError, and checks a false property\n"
            "    assert 2 + 3 == 6\n"
            "    return True\n"
        ),
    },
    {
        "id": "t09",
        "task_type": "test_generation",
        "entry_func": "test_string_reverse_roundtrip",
        "prompt": (
            "Write a function named `test_string_reverse_roundtrip` that takes no "
            "arguments. For at least 4 different strings, assert that reversing a "
            "string twice (using slicing, e.g. `s[::-1][::-1]`) returns the original "
            "string. Catch any `AssertionError` internally and return `True` if all "
            "checks passed, `False` otherwise."
        ),
        "tests": [
            "assert test_string_reverse_roundtrip() is True",
            "assert callable(test_string_reverse_roundtrip)",
        ],
        "reference_solution": (
            "def test_string_reverse_roundtrip():\n"
            "    try:\n"
            "        for s in ['hello', '', 'a', 'racecar', 'CodeShift 2026']:\n"
            "            assert s[::-1][::-1] == s\n"
            "        return True\n"
            "    except AssertionError:\n"
            "        return False\n"
        ),
        "buggy_solution": (
            "def test_string_reverse_roundtrip():\n"
            "    # bug: compares against the single-reversed string, always False\n"
            "    try:\n"
            "        for s in ['hello', 'world']:\n"
            "            assert s[::-1] == s\n"
            "        return True\n"
            "    except AssertionError:\n"
            "        return False\n"
        ),
    },
    {
        "id": "t10",
        "task_type": "test_generation",
        "entry_func": "test_list_append_contract",
        "prompt": (
            "Write a function named `test_list_append_contract` that takes no "
            "arguments and asserts, for at least 3 different starting lists, that "
            "calling `.append(x)` increases the list's length by exactly 1 and that "
            "the appended element becomes the last element. Catch any "
            "`AssertionError` internally and return `True`/`False` accordingly."
        ),
        "tests": [
            "assert test_list_append_contract() is True",
            "assert callable(test_list_append_contract)",
        ],
        "reference_solution": (
            "def test_list_append_contract():\n"
            "    try:\n"
            "        for start in [[], [1], [1, 2, 3]]:\n"
            "            before_len = len(start)\n"
            "            start.append('x')\n"
            "            assert len(start) == before_len + 1\n"
            "            assert start[-1] == 'x'\n"
            "        return True\n"
            "    except AssertionError:\n"
            "        return False\n"
        ),
        "buggy_solution": (
            "def test_list_append_contract():\n"
            "    # bug: checks length stays the same (wrong contract)\n"
            "    try:\n"
            "        for start in [[], [1]]:\n"
            "            before_len = len(start)\n"
            "            start.append('x')\n"
            "            assert len(start) == before_len\n"
            "        return True\n"
            "    except AssertionError:\n"
            "        return False\n"
        ),
    },
    # ------------------------------------------------------------------------- refactor
    {
        "id": "t11",
        "task_type": "refactor",
        "entry_func": "max_of",
        "prompt": (
            "These three near-duplicate functions each find the maximum of a list "
            "using a hand-rolled loop. Refactor them into a single function "
            "`max_of(nums)` that raises `ValueError` on an empty list (matching "
            "Python's built-in `max` behavior).\n\n"
            "```python\n"
            "def max_of_ints(nums):\n"
            "    m = nums[0]\n"
            "    for n in nums:\n"
            "        if n > m:\n"
            "            m = n\n"
            "    return m\n"
            "\n"
            "def max_of_floats(nums):\n"
            "    m = nums[0]\n"
            "    for n in nums:\n"
            "        if n > m:\n"
            "            m = n\n"
            "    return m\n"
            "```"
        ),
        "tests": [
            "assert max_of([3, 1, 2]) == 3",
            "assert max_of([-5, -1, -9]) == -1",
            (
                "try:\n"
                "    max_of([])\n"
                "    assert False, 'expected ValueError'\n"
                "except ValueError:\n"
                "    pass"
            ),
        ],
        "reference_solution": "def max_of(nums):\n    return max(nums)\n",
        "buggy_solution": (
            "def max_of(nums):\n"
            "    # bug: returns None on empty list instead of raising\n"
            "    if not nums:\n"
            "        return None\n"
            "    m = nums[0]\n"
            "    for n in nums:\n"
            "        if n > m:\n"
            "            m = n\n"
            "    return m\n"
        ),
    },
    {
        "id": "t12",
        "task_type": "refactor",
        "entry_func": "age_group",
        "prompt": (
            "Refactor this deeply nested if/else pyramid into a clean, flat "
            "function `age_group(age)` returning one of 'child', 'teen', 'adult', "
            "'senior'.\n\n"
            "```python\n"
            "def age_group(age):\n"
            "    if age < 65:\n"
            "        if age < 20:\n"
            "            if age < 13:\n"
            "                return 'child'\n"
            "            else:\n"
            "                return 'teen'\n"
            "        else:\n"
            "            return 'adult'\n"
            "    else:\n"
            "        return 'senior'\n"
            "```"
        ),
        "tests": [
            "assert age_group(5) == 'child'",
            "assert age_group(15) == 'teen'",
            "assert age_group(40) == 'adult'",
            "assert age_group(70) == 'senior'",
        ],
        "reference_solution": (
            "def age_group(age):\n"
            "    if age < 13:\n"
            "        return 'child'\n"
            "    elif age < 20:\n"
            "        return 'teen'\n"
            "    elif age < 65:\n"
            "        return 'adult'\n"
            "    else:\n"
            "        return 'senior'\n"
        ),
        "buggy_solution": (
            "def age_group(age):\n"
            "    # bug: teen/child boundary shifted\n"
            "    if age < 20:\n"
            "        return 'child'\n"
            "    elif age < 65:\n"
            "        return 'adult'\n"
            "    else:\n"
            "        return 'senior'\n"
        ),
    },
    {
        "id": "t13",
        "task_type": "refactor",
        "entry_func": "join_words",
        "prompt": (
            "This function builds a string with inefficient repeated "
            "concatenation in a loop. Refactor it into `join_words(words, sep)` "
            "using idiomatic `str.join`.\n\n"
            "```python\n"
            "def join_words(words, sep):\n"
            "    result = ''\n"
            "    for i, w in enumerate(words):\n"
            "        result += w\n"
            "        if i < len(words) - 1:\n"
            "            result += sep\n"
            "    return result\n"
            "```"
        ),
        "tests": [
            "assert join_words(['a', 'b', 'c'], '-') == 'a-b-c'",
            "assert join_words([], ',') == ''",
            "assert join_words(['solo'], ',') == 'solo'",
        ],
        "reference_solution": "def join_words(words, sep):\n    return sep.join(words)\n",
        "buggy_solution": (
            "def join_words(words, sep):\n"
            "    # bug: trailing separator\n"
            "    result = ''\n"
            "    for w in words:\n"
            "        result += w + sep\n"
            "    return result\n"
        ),
    },
    # --------------------------------------------------------------------- feature_build
    {
        "id": "t14",
        "task_type": "feature_build",
        "entry_func": "word_frequencies",
        "prompt": (
            "Implement a new function `word_frequencies(text)` that returns a dict "
            "mapping each lowercase word to how many times it appears in `text`. "
            "Ignore punctuation and be case-insensitive."
        ),
        "tests": [
            "assert word_frequencies('Hi hi HI!') == {'hi': 3}",
            "assert word_frequencies('') == {}",
            "assert word_frequencies('one two two three three three') == {'one': 1, 'two': 2, 'three': 3}",
        ],
        "reference_solution": (
            "import re\n\n"
            "def word_frequencies(text):\n"
            "    words = re.findall(r\"[a-zA-Z']+\", text.lower())\n"
            "    freq = {}\n"
            "    for w in words:\n"
            "        freq[w] = freq.get(w, 0) + 1\n"
            "    return freq\n"
        ),
        "buggy_solution": (
            "def word_frequencies(text):\n"
            "    # bug: does not lowercase, so 'Hi' and 'hi' are counted separately\n"
            "    words = text.replace('!', '').split()\n"
            "    freq = {}\n"
            "    for w in words:\n"
            "        freq[w] = freq.get(w, 0) + 1\n"
            "    return freq\n"
        ),
    },
    {
        "id": "t15",
        "task_type": "feature_build",
        "entry_func": "flatten",
        "prompt": (
            "Implement a new function `flatten(nested_list)` that flattens an "
            "arbitrarily deeply nested list of lists into a single flat list, "
            "preserving element order."
        ),
        "tests": [
            "assert flatten([1, [2, 3], [4, [5, 6]]]) == [1, 2, 3, 4, 5, 6]",
            "assert flatten([]) == []",
            "assert flatten([1, 2, 3]) == [1, 2, 3]",
            "assert flatten([[[[1]]], 2]) == [1, 2]",
        ],
        "reference_solution": (
            "def flatten(nested_list):\n"
            "    result = []\n"
            "    for item in nested_list:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
        "buggy_solution": (
            "def flatten(nested_list):\n"
            "    # bug: only flattens one level deep\n"
            "    result = []\n"
            "    for item in nested_list:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(item)\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result\n"
        ),
    },
]


def get_task(task_id: str) -> dict:
    """Look up a single task by id, raising KeyError with a helpful message."""
    for t in TASKS:
        if t["id"] == task_id:
            return t
    raise KeyError(f"No task with id {task_id!r}. Known ids: {[t['id'] for t in TASKS]}")


if __name__ == "__main__":
    # Quick self-check: every reference_solution must pass its own hidden tests,
    # and every buggy_solution must fail at least one (sanity check on task design).
    for task in TASKS:
        ns: dict = {}
        exec(compile(task["reference_solution"], "<reference>", "exec"), ns)  # nosec B102 - author-written reference code in this file, never model output
        for test in task["tests"]:
            exec(compile(test, "<test>", "exec"), dict(ns))  # nosec B102 - author-written test from this file

        ns_bug: dict = {}
        exec(compile(task["buggy_solution"], "<buggy>", "exec"), ns_bug)  # nosec B102 - author-written buggy fixture from this file
        failed = False
        for test in task["tests"]:
            try:
                exec(compile(test, "<test>", "exec"), dict(ns_bug))  # nosec B102 - author-written test from this file
            except Exception:
                failed = True
        assert failed, f"buggy_solution for {task['id']} unexpectedly passed all tests"
    print(f"OK: {len(TASKS)} tasks self-checked (reference passes, buggy fails).")

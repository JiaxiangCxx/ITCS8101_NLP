# Search and CSP exercises

Two classic AI search assignments, in plain Python with no dependencies.

## `ai.py`: N-Queens with hill climbing

Local search for the N-Queens problem. The state stores one queen per column; the
heuristic is the number of attacking pairs.

- steepest-ascent hill climbing
- hill climbing with sideways moves (limit 100)
- random-restart versions of both
- experiments over 50 to 1500 trials that report success rate and average steps

```bash
python ai.py        # prompts for N (default 8)
```

## `csp.py`: Map coloring as a CSP

Finds the chromatic number of the Australia and USA maps with backtracking search,
and compares these variants:

| Method | Description |
|---|---|
| DFS | plain backtracking |
| DFS+FC | with forward checking |
| DFS+FC+PROP | with forward checking and singleton propagation |
| … +Heuristics | MRV / degree variable ordering and LCV value ordering |

```bash
python csp.py       # writes map_coloring_results.csv
```

`map_coloring_results.csv` holds the results: chromatic number, number of
backtracks and runtime per map, method and trial.

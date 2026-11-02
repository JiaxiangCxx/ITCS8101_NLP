import time
import random
import copy
import csv
import signal


# =====================================================
# Timeout support (Linux / cluster)
# =====================================================

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Method timed out")


signal.signal(signal.SIGALRM, timeout_handler)


# =====================================================
# Map definitions
# =====================================================

AUSTRALIA_MAP = {
    "WA": ["NT", "SA"],
    "NT": ["WA", "SA", "Q"],
    "SA": ["WA", "NT", "Q", "NSW", "V"],
    "Q": ["NT", "SA", "NSW"],
    "NSW": ["Q", "SA", "V"],
    "V": ["SA", "NSW"],
    "T": []
}

USA_MAP = {
    "AL": ["FL", "GA", "MS", "TN"],
    "AK": [],
    "AZ": ["CA", "NV", "UT", "NM"],
    "AR": ["MO", "TN", "MS", "LA", "TX", "OK"],
    "CA": ["OR", "NV", "AZ"],
    "CO": ["WY", "NE", "KS", "OK", "NM", "UT"],
    "CT": ["NY", "MA", "RI"],
    "DE": ["MD", "NJ", "PA"],
    "FL": ["AL", "GA"],
    "GA": ["FL", "AL", "TN", "NC", "SC"],
    "HI": [],
    "ID": ["WA", "OR", "NV", "UT", "WY", "MT"],
    "IL": ["WI", "IA", "MO", "KY", "IN"],
    "IN": ["MI", "OH", "KY", "IL"],
    "IA": ["MN", "SD", "NE", "MO", "IL", "WI"],
    "KS": ["NE", "MO", "OK", "CO"],
    "KY": ["IL", "IN", "OH", "WV", "VA", "TN", "MO"],
    "LA": ["TX", "AR", "MS"],
    "ME": ["NH"],
    "MD": ["VA", "WV", "PA", "DE"],
    "MA": ["RI", "CT", "NY", "VT", "NH"],
    "MI": ["WI", "IN", "OH"],
    "MN": ["ND", "SD", "IA", "WI"],
    "MS": ["LA", "AR", "TN", "AL"],
    "MO": ["IA", "IL", "KY", "TN", "AR", "OK", "KS", "NE"],
    "MT": ["ID", "WY", "SD", "ND"],
    "NE": ["SD", "IA", "MO", "KS", "CO", "WY"],
    "NV": ["OR", "ID", "UT", "AZ", "CA"],
    "NH": ["ME", "MA", "VT"],
    "NJ": ["NY", "DE", "PA"],
    "NM": ["AZ", "UT", "CO", "OK", "TX"],
    "NY": ["PA", "NJ", "CT", "MA", "VT"],
    "NC": ["VA", "TN", "GA", "SC"],
    "ND": ["MT", "SD", "MN"],
    "OH": ["MI", "PA", "WV", "KY", "IN"],
    "OK": ["CO", "KS", "MO", "AR", "TX", "NM"],
    "OR": ["WA", "ID", "NV", "CA"],
    "PA": ["NY", "NJ", "DE", "MD", "WV", "OH"],
    "RI": ["CT", "MA"],
    "SC": ["NC", "GA"],
    "SD": ["ND", "MN", "IA", "NE", "WY", "MT"],
    "TN": ["KY", "VA", "NC", "GA", "AL", "MS", "AR", "MO"],
    "TX": ["NM", "OK", "AR", "LA"],
    "UT": ["ID", "WY", "CO", "NM", "AZ", "NV"],
    "VT": ["NY", "NH", "MA"],
    "VA": ["NC", "TN", "KY", "WV", "MD"],
    "WA": ["ID", "OR"],
    "WV": ["OH", "PA", "MD", "VA", "KY"],
    "WI": ["MN", "IA", "IL", "MI"],
    "WY": ["MT", "SD", "NE", "CO", "UT", "ID"]
}


# =====================================================
# Utility functions
# =====================================================

def is_consistent(var, value, assignment, graph):
    for neighbor in graph[var]:
        if neighbor in assignment and assignment[neighbor] == value:
            return False
    return True


def get_random_variable_order(graph, seed=None):
    vars_list = list(graph.keys())
    rng = random.Random(seed)
    rng.shuffle(vars_list)
    return vars_list


def select_unassigned_variable(assignment, domains, graph, use_heuristics=False, fixed_order=None):
    unassigned = [v for v in graph if v not in assignment]

    if not unassigned:
        return None

    if not use_heuristics:
        if fixed_order is not None:
            for var in fixed_order:
                if var not in assignment:
                    return var
        return unassigned[0]

    # MRV
    min_domain_size = min(len(domains[v]) for v in unassigned)
    mrv_vars = [v for v in unassigned if len(domains[v]) == min_domain_size]

    if len(mrv_vars) == 1:
        return mrv_vars[0]

    # Degree constraint tie-breaker
    def unassigned_degree(var):
        return sum(1 for n in graph[var] if n not in assignment)

    return max(mrv_vars, key=unassigned_degree)


def order_domain_values(var, assignment, domains, graph, use_heuristics=False):
    values = list(domains[var])

    if not use_heuristics:
        return values

    # LCV
    def ruled_out_count(value):
        count = 0
        for neighbor in graph[var]:
            if neighbor not in assignment and value in domains[neighbor]:
                count += 1
        return count

    return sorted(values, key=ruled_out_count)


def forward_check(var, value, domains, assignment, graph):
    new_domains = copy.deepcopy(domains)

    for neighbor in graph[var]:
        if neighbor not in assignment and value in new_domains[neighbor]:
            new_domains[neighbor].remove(value)
            if len(new_domains[neighbor]) == 0:
                return False, new_domains

    return True, new_domains


def propagate_singletons(domains, assignment, graph):
    new_assignment = assignment.copy()
    new_domains = copy.deepcopy(domains)

    changed = True
    while changed:
        changed = False

        singleton_vars = [
            v for v in graph
            if v not in new_assignment and len(new_domains[v]) == 1
        ]

        for var in singleton_vars:
            value = next(iter(new_domains[var]))

            if not is_consistent(var, value, new_assignment, graph):
                return False, new_assignment, new_domains

            new_assignment[var] = value
            new_domains[var] = {value}
            changed = True

            ok, new_domains = forward_check(var, value, new_domains, new_assignment, graph)
            if not ok:
                return False, new_assignment, new_domains

    return True, new_assignment, new_domains


# =====================================================
# Backtracking solver
# =====================================================

def backtrack(assignment, domains, graph, stats,
              use_fc=False, use_propagation=False,
              use_heuristics=False, fixed_order=None):
    if len(assignment) == len(graph):
        return assignment

    var = select_unassigned_variable(
        assignment, domains, graph,
        use_heuristics=use_heuristics,
        fixed_order=fixed_order
    )

    if var is None:
        return assignment

    for value in order_domain_values(var, assignment, domains, graph, use_heuristics):
        if is_consistent(var, value, assignment, graph):
            new_assignment = assignment.copy()
            new_assignment[var] = value
            new_domains = copy.deepcopy(domains)
            new_domains[var] = {value}

            if use_fc:
                ok, new_domains = forward_check(var, value, new_domains, new_assignment, graph)
                if not ok:
                    stats["backtracks"] += 1
                    continue

            if use_propagation:
                ok, new_assignment, new_domains = propagate_singletons(
                    new_domains, new_assignment, graph
                )
                if not ok:
                    stats["backtracks"] += 1
                    continue

            result = backtrack(
                new_assignment,
                new_domains,
                graph,
                stats,
                use_fc=use_fc,
                use_propagation=use_propagation,
                use_heuristics=use_heuristics,
                fixed_order=fixed_order
            )

            if result is not None:
                return result

    stats["backtracks"] += 1
    return None


# =====================================================
# Solve with k colors
# =====================================================

def solve_map(graph, num_colors, use_fc=False, use_propagation=False,
              use_heuristics=False, fixed_order=None):
    colors = set(range(1, num_colors + 1))
    domains = {var: set(colors) for var in graph}
    stats = {"backtracks": 0}

    start = time.perf_counter()
    solution = backtrack(
        assignment={},
        domains=domains,
        graph=graph,
        stats=stats,
        use_fc=use_fc,
        use_propagation=use_propagation,
        use_heuristics=use_heuristics,
        fixed_order=fixed_order
    )
    end = time.perf_counter()

    return solution, stats["backtracks"], end - start


def lower_bound_colors(graph):
    if len(graph) <= 10:
        return 3
    return 4


def find_chromatic_number(graph, use_fc=False, use_propagation=False,
                          use_heuristics=False, fixed_order=None):
    start_k = lower_bound_colors(graph)

    for k in range(start_k, len(graph) + 1):
        solution, backtracks, elapsed = solve_map(
            graph,
            k,
            use_fc=use_fc,
            use_propagation=use_propagation,
            use_heuristics=use_heuristics,
            fixed_order=fixed_order
        )
        if solution is not None:
            return k, solution, backtracks, elapsed

    return None, None, None, None


def run_with_timeout(timeout_seconds, graph, use_fc, use_prop, use_heur, fixed_order):
    signal.alarm(timeout_seconds)
    try:
        return find_chromatic_number(
            graph,
            use_fc=use_fc,
            use_propagation=use_prop,
            use_heuristics=use_heur,
            fixed_order=fixed_order
        )
    finally:
        signal.alarm(0)


# =====================================================
# Main experiment runner
# =====================================================

def main():
    filename = "map_coloring_results.csv"

    fieldnames = [
        "map",
        "trial",
        "method",
        "chromatic_number",
        "backtracks",
        "time_seconds",
        "status"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    methods = [
        ("DFS", False, False, False),
        ("DFS+FC", True, False, False),
        ("DFS+FC+PROP", True, True, False),
        ("DFS+Heuristics", False, False, True),
        ("DFS+FC+Heuristics", True, False, True),
        ("DFS+FC+PROP+Heuristics", True, True, True),
    ]

    all_results = []

    for map_name, graph in [("Australia", AUSTRALIA_MAP), ("USA", USA_MAP)]:
        for trial in range(1, 6):
            print(f"\n========== {map_name} | Trial {trial}/5 ==========")

            fixed_order = get_random_variable_order(graph, seed=trial)
            print(f"Random order for non-heuristic methods: {fixed_order}")

            for method_name, use_fc, use_prop, use_heur in methods:
                print(f"\nRunning: {method_name}")

                order_used = None if use_heur else fixed_order

                timeout_seconds = None
                if map_name == "USA" and method_name == "DFS":
                    timeout_seconds = 300
                elif map_name == "USA" and method_name == "DFS+FC":
                    timeout_seconds = 180

                try:
                    if timeout_seconds is not None:
                        chromatic_number, solution, backtracks, elapsed = run_with_timeout(
                            timeout_seconds, graph, use_fc, use_prop, use_heur, order_used
                        )
                    else:
                        chromatic_number, solution, backtracks, elapsed = find_chromatic_number(
                            graph,
                            use_fc=use_fc,
                            use_propagation=use_prop,
                            use_heuristics=use_heur,
                            fixed_order=order_used
                        )

                    row = {
                        "map": map_name,
                        "trial": trial,
                        "method": method_name,
                        "chromatic_number": chromatic_number,
                        "backtracks": backtracks,
                        "time_seconds": elapsed,
                        "status": "OK"
                    }

                    print(
                        f"Result: Chromatic={chromatic_number}, "
                        f"Backtracks={backtracks}, Time={elapsed:.6f}s"
                    )

                except TimeoutException:
                    row = {
                        "map": map_name,
                        "trial": trial,
                        "method": method_name,
                        "chromatic_number": "TIMEOUT",
                        "backtracks": "TIMEOUT",
                        "time_seconds": timeout_seconds,
                        "status": "TIMEOUT"
                    }

                    print(f"Result: TIMEOUT after {timeout_seconds} seconds")

                all_results.append(row)

                with open(filename, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(row)

    print(f"\nResults saved to {filename}")


if __name__ == "__main__":
    main()
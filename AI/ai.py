import random
from typing import List, Tuple, Dict, Optional

# =========================
# Global settings
# =========================
DEFAULT_N = 8
SIDEWAYS_LIMIT = 100
TRIAL_LIST = [50, 100, 200, 500, 1000, 1500]


# =========================
# Basic utility functions
# =========================
def random_state(n: int) -> List[int]:
    """Generate a random state.
    Index = column, value = row of queen in that column.
    """
    return [random.randint(0, n - 1) for _ in range(n)]


def heuristic(state: List[int]) -> int:
    """Return number of attacking pairs of queens."""
    h = 0
    n = len(state)

    for c1 in range(n):
        for c2 in range(c1 + 1, n):
            r1, r2 = state[c1], state[c2]

            # same row
            if r1 == r2:
                h += 1
            # same diagonal
            elif abs(r1 - r2) == abs(c1 - c2):
                h += 1

    return h


def is_goal(state: List[int]) -> bool:
    return heuristic(state) == 0


def get_neighbors(state: List[int]) -> List[List[int]]:
    """Generate all neighbors by moving one queen within its column."""
    neighbors = []
    n = len(state)

    for col in range(n):
        current_row = state[col]
        for row in range(n):
            if row != current_row:
                neighbor = state.copy()
                neighbor[col] = row
                neighbors.append(neighbor)

    return neighbors


def choose_best_neighbor(state: List[int]) -> Tuple[List[List[int]], int]:
    """Return all best neighbors and their best heuristic."""
    neighbors = get_neighbors(state)
    best_h = min(heuristic(nei) for nei in neighbors)
    best_neighbors = [nei for nei in neighbors if heuristic(nei) == best_h]
    return best_neighbors, best_h


def state_to_string(state: List[int]) -> str:
    return str(state)


def board_to_string(state: List[int]) -> str:
    """Optional pretty board display."""
    n = len(state)
    lines = []
    for r in range(n):
        row = []
        for c in range(n):
            if state[c] == r:
                row.append("Q")
            else:
                row.append(".")
        lines.append(" ".join(row))
    return "\n".join(lines)


# =========================
# Hill climbing algorithms
# =========================
def hill_climbing(start_state: List[int]) -> Dict:
    """Standard hill climbing without sideways moves."""
    current = start_state.copy()
    steps = 0
    sequence = [(current.copy(), heuristic(current))]

    while True:
        current_h = heuristic(current)
        if current_h == 0:
            return {
                "success": True,
                "steps": steps,
                "final_state": current,
                "sequence": sequence,
            }

        best_neighbors, best_h = choose_best_neighbor(current)

        # no improvement => fail
        if best_h >= current_h:
            return {
                "success": False,
                "steps": steps,
                "final_state": current,
                "sequence": sequence,
            }

        current = random.choice(best_neighbors)
        steps += 1
        sequence.append((current.copy(), heuristic(current)))


def hill_climbing_sideways(start_state: List[int], sideways_limit: int = SIDEWAYS_LIMIT) -> Dict:
    """Hill climbing with sideways moves allowed up to sideways_limit."""
    current = start_state.copy()
    steps = 0
    sideways_used = 0
    sequence = [(current.copy(), heuristic(current))]

    while True:
        current_h = heuristic(current)
        if current_h == 0:
            return {
                "success": True,
                "steps": steps,
                "final_state": current,
                "sequence": sequence,
                "sideways_used": sideways_used,
            }

        best_neighbors, best_h = choose_best_neighbor(current)

        if best_h < current_h:
            current = random.choice(best_neighbors)
            steps += 1
            sideways_used = 0
            sequence.append((current.copy(), heuristic(current)))

        elif best_h == current_h and sideways_used < sideways_limit:
            current = random.choice(best_neighbors)
            steps += 1
            sideways_used += 1
            sequence.append((current.copy(), heuristic(current)))

        else:
            return {
                "success": False,
                "steps": steps,
                "final_state": current,
                "sequence": sequence,
                "sideways_used": sideways_used,
            }


# =========================
# Random restart versions
# =========================
def random_restart_hill_climbing(n: int) -> Dict:
    """Random-restart hill climbing without sideways moves."""
    total_steps = 0
    restarts = 0

    while True:
        start = random_state(n)
        result = hill_climbing(start)
        total_steps += result["steps"]

        if result["success"]:
            return {
                "success": True,
                "restarts": restarts,
                "steps": total_steps,
                "final_state": result["final_state"],
            }

        restarts += 1


def random_restart_hill_climbing_sideways(n: int, sideways_limit: int = SIDEWAYS_LIMIT) -> Dict:
    """Random-restart hill climbing with sideways moves."""
    total_steps = 0
    restarts = 0

    while True:
        start = random_state(n)
        result = hill_climbing_sideways(start, sideways_limit=sideways_limit)
        total_steps += result["steps"]

        if result["success"]:
            return {
                "success": True,
                "restarts": restarts,
                "steps": total_steps,
                "final_state": result["final_state"],
            }

        restarts += 1


# =========================
# Experiment functions
# =========================
def run_trials_hill_climbing(n: int, num_trials: int) -> Dict:
    success_count = 0
    failure_count = 0
    success_steps = []
    failure_steps = []

    for _ in range(num_trials):
        start = random_state(n)
        result = hill_climbing(start)

        if result["success"]:
            success_count += 1
            success_steps.append(result["steps"])
        else:
            failure_count += 1
            failure_steps.append(result["steps"])

    return {
        "trials": num_trials,
        "success_rate": success_count / num_trials,
        "failure_rate": failure_count / num_trials,
        "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0,
        "avg_steps_failure": sum(failure_steps) / len(failure_steps) if failure_steps else 0,
    }


def run_trials_hill_climbing_sideways(n: int, num_trials: int, sideways_limit: int = SIDEWAYS_LIMIT) -> Dict:
    success_count = 0
    failure_count = 0
    success_steps = []
    failure_steps = []

    for _ in range(num_trials):
        start = random_state(n)
        result = hill_climbing_sideways(start, sideways_limit=sideways_limit)

        if result["success"]:
            success_count += 1
            success_steps.append(result["steps"])
        else:
            failure_count += 1
            failure_steps.append(result["steps"])

    return {
        "trials": num_trials,
        "success_rate": success_count / num_trials,
        "failure_rate": failure_count / num_trials,
        "avg_steps_success": sum(success_steps) / len(success_steps) if success_steps else 0,
        "avg_steps_failure": sum(failure_steps) / len(failure_steps) if failure_steps else 0,
    }


def run_trials_random_restart(n: int, num_trials: int, sideways: bool = False, sideways_limit: int = SIDEWAYS_LIMIT) -> Dict:
    restart_counts = []
    step_counts = []

    for _ in range(num_trials):
        if sideways:
            result = random_restart_hill_climbing_sideways(n, sideways_limit=sideways_limit)
        else:
            result = random_restart_hill_climbing(n)

        restart_counts.append(result["restarts"])
        step_counts.append(result["steps"])

    return {
        "trials": num_trials,
        "avg_restarts": sum(restart_counts) / len(restart_counts),
        "avg_steps": sum(step_counts) / len(step_counts),
    }


# =========================
# Sequence printing
# =========================
def print_search_sequence(result: Dict, title: str = ""):
    if title:
        print(title)

    for i, (state, h) in enumerate(result["sequence"]):
        if i == 0:
            print(f"Initial: {state}  h={h}")
        else:
            print(f"Step {i}:   {state}  h={h}")

    print("Success" if result["success"] else "Failure")
    print("-" * 60)


def generate_four_sequences(n: int, sideways: bool = False, sideways_limit: int = SIDEWAYS_LIMIT):
    print("\n" + "=" * 60)
    print("FOUR RANDOM SEARCH SEQUENCES")
    print("=" * 60)

    for i in range(4):
        start = random_state(n)
        if sideways:
            result = hill_climbing_sideways(start, sideways_limit=sideways_limit)
            print_search_sequence(result, title=f"Sequence {i+1} (with sideways)")
        else:
            result = hill_climbing(start)
            print_search_sequence(result, title=f"Sequence {i+1} (without sideways)")


# =========================
# Reporting helpers
# =========================
def print_hill_table(results: List[Dict], title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{'Trials':<10}{'Success Rate':<18}{'Failure Rate':<18}{'Avg Steps Success':<20}{'Avg Steps Failure':<20}")
    for r in results:
        print(
            f"{r['trials']:<10}"
            f"{r['success_rate']:<18.4f}"
            f"{r['failure_rate']:<18.4f}"
            f"{r['avg_steps_success']:<20.4f}"
            f"{r['avg_steps_failure']:<20.4f}"
        )


def print_rr_table(results: List[Dict], title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(f"{'Trials':<10}{'Avg Restarts':<20}{'Avg Steps':<20}")
    for r in results:
        print(
            f"{r['trials']:<10}"
            f"{r['avg_restarts']:<20.4f}"
            f"{r['avg_steps']:<20.4f}"
        )


# =========================
# Main experiment driver
# =========================
def main():
    print("N-Queens using Hill Climbing and Variants")
    try:
        n = int(input(f"Enter number of queens N (default {DEFAULT_N}): ").strip() or DEFAULT_N)
    except ValueError:
        n = DEFAULT_N

    print(f"\nUsing N = {n}")
    print(f"Sideways move limit = {SIDEWAYS_LIMIT}")

    # A. Hill climbing
    hc_results = []
    for trials in TRIAL_LIST:
        hc_results.append(run_trials_hill_climbing(n, trials))
    print_hill_table(hc_results, "A. Hill Climbing Search")

    # Search sequences from four random initial configurations
    generate_four_sequences(n, sideways=False)

    # B. Hill climbing with sideways move
    hcs_results = []
    for trials in TRIAL_LIST:
        hcs_results.append(run_trials_hill_climbing_sideways(n, trials, sideways_limit=SIDEWAYS_LIMIT))
    print_hill_table(hcs_results, "B. Hill Climbing Search with Sideways Move")

    # Search sequences from four random initial configurations
    generate_four_sequences(n, sideways=True, sideways_limit=SIDEWAYS_LIMIT)

    # C. Random-restart hill climbing
    rr_no_sideways_results = []
    rr_sideways_results = []

    for trials in TRIAL_LIST:
        rr_no_sideways_results.append(run_trials_random_restart(n, trials, sideways=False))
        rr_sideways_results.append(run_trials_random_restart(n, trials, sideways=True, sideways_limit=SIDEWAYS_LIMIT))

    print_rr_table(rr_no_sideways_results, "C1. Random-Restart Hill Climbing without Sideways Move")
    print_rr_table(rr_sideways_results, "C2. Random-Restart Hill Climbing with Sideways Move")


if __name__ == "__main__":
    main()
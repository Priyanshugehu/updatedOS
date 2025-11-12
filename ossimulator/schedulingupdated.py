# ================================================================
# 🧠 OSync OS Algorithm Simulation Functions
# Contains implementations of Page Replacement and CPU Scheduling
# algorithms to run after ML prediction.
# ================================================================

import numpy as np

# ================================================================
# 🔹 PAGE REPLACEMENT ALGORITHMS
# ================================================================

def fifo_page_replacement(pages, frames):
    """Simulate FIFO Page Replacement"""
    memory = []
    page_faults = 0
    for page in pages:
        if page not in memory:
            if len(memory) < frames:
                memory.append(page)
            else:
                memory.pop(0)
                memory.append(page)
            page_faults += 1
    return page_faults


def lru_page_replacement(pages, frames):
    """Simulate LRU Page Replacement"""
    memory = []
    page_faults = 0
    recently_used = []
    for page in pages:
        if page not in memory:
            if len(memory) < frames:
                memory.append(page)
            else:
                lru_page = recently_used.pop(0)
                memory.remove(lru_page)
                memory.append(page)
            page_faults += 1
        if page in recently_used:
            recently_used.remove(page)
        recently_used.append(page)
    return page_faults


def optimal_page_replacement(pages, frames):
    """Simulate Optimal Page Replacement"""
    memory = []
    page_faults = 0
    for i in range(len(pages)):
        if pages[i] not in memory:
            if len(memory) < frames:
                memory.append(pages[i])
            else:
                future = pages[i + 1:]
                indices = []
                for x in memory:
                    if x in future:
                        indices.append(future.index(x))
                    else:
                        indices.append(float('inf'))
                memory.pop(indices.index(max(indices)))
                memory.append(pages[i])
            page_faults += 1
    return page_faults


# ================================================================
# ⚙️ CPU SCHEDULING ALGORITHMS
# ================================================================

def fcfs_scheduling(processes, burst_times):
    """Simulate First Come First Serve Scheduling"""
    waiting_times = [0]
    for i in range(1, len(burst_times)):
        waiting_times.append(waiting_times[-1] + burst_times[i - 1])
    avg_waiting = sum(waiting_times) / len(waiting_times)
    turnaround_times = [wt + bt for wt, bt in zip(waiting_times, burst_times)]
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


def sjf_scheduling(processes, burst_times):
    """Simulate Shortest Job First (Non-Preemptive) Scheduling"""
    sorted_indices = np.argsort(burst_times)
    burst_sorted = [burst_times[i] for i in sorted_indices]
    waiting_times = [0]
    for i in range(1, len(burst_sorted)):
        waiting_times.append(waiting_times[-1] + burst_sorted[i - 1])
    avg_waiting = sum(waiting_times) / len(waiting_times)
    turnaround_times = [wt + bt for wt, bt in zip(waiting_times, burst_sorted)]
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Order": [processes[i] for i in sorted_indices],
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


def round_robin_scheduling(processes, burst_times, quantum):
    """Simulate Round Robin Scheduling"""
    remaining_times = burst_times[:]
    waiting_times = [0] * len(burst_times)
    turnaround_times = [0] * len(burst_times)
    t = 0
    done = False

    while not done:
        done = True
        for i in range(len(burst_times)):
            if remaining_times[i] > 0:
                done = False
                if remaining_times[i] > quantum:
                    t += quantum
                    remaining_times[i] -= quantum
                else:
                    t += remaining_times[i]
                    waiting_times[i] = t - burst_times[i]
                    remaining_times[i] = 0

    for i in range(len(burst_times)):
        turnaround_times[i] = burst_times[i] + waiting_times[i]

    avg_waiting = sum(waiting_times) / len(waiting_times)
    avg_turnaround = sum(turnaround_times) / len(turnaround_times)
    return {
        "Waiting Times": waiting_times,
        "Turnaround Times": turnaround_times,
        "Avg Waiting Time": round(avg_waiting, 2),
        "Avg Turnaround Time": round(avg_turnaround, 2)
    }


# ================================================================
# 🧩 WRAPPER FUNCTIONS
# ================================================================

def simulate_page_replacement(algo_name, pages, frames):
    """Run the selected Page Replacement Algorithm"""
    algo_name = algo_name.lower()
    if algo_name == "fifo":
        faults = fifo_page_replacement(pages, frames)
    elif algo_name == "lru":
        faults = lru_page_replacement(pages, frames)
    elif algo_name == "optimal":
        faults = optimal_page_replacement(pages, frames)
    else:
        raise ValueError("Unknown Page Replacement Algorithm")
    return {"Page Faults": faults, "Algorithm": algo_name.upper()}


def simulate_cpu_scheduling(algo_name, processes, burst_times, quantum=None):
    """Run the selected CPU Scheduling Algorithm"""
    algo_name = algo_name.lower()
    if algo_name == "fcfs":
        return fcfs_scheduling(processes, burst_times)
    elif algo_name == "sjf":
        return sjf_scheduling(processes, burst_times)
    elif algo_name in ["rr", "round robin", "roundrobin"]:
        if quantum is None:
            raise ValueError("Quantum required for Round Robin")
        return round_robin_scheduling(processes, burst_times, quantum)
    else:
        raise ValueError("Unknown Scheduling Algorithm")

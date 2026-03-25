"""Simulation engine with five routing strategies."""

import math
import random

from orbital_model import SimMetrics, build_satellites, next_comm_entry


# ---------------------------------------------------------------------------
# Strategy decision functions
# ---------------------------------------------------------------------------

def _decide_static(proc_idx, comm_ids):
    """Fixed round-robin: processor index i -> communicator i % Y."""
    return "forward", comm_ids[proc_idx % len(comm_ids)]


def _decide_greedy_evf(comm_states, t, T_comp, T_idle, T_orbit):
    """Pick communicator with earliest comm window, ignoring load / energy."""
    best_id, best_wait = None, float("inf")
    for c_id, _, _, c_offset in comm_states:
        orbit_clock = (t + c_offset) % T_orbit
        if orbit_clock >= T_comp + T_idle:
            continue
        wait = next_comm_entry(t, c_offset, T_comp, T_idle, T_orbit) - t
        if wait < best_wait:
            best_wait = wait
            best_id = c_id
    if best_id is not None:
        return "forward", best_id
    return "direct", None


def _decide_lia(comm_states, t, proc_offset, D, R_max, q, e_p,
                T_comp, T_idle, T_comm, T_orbit):
    """Full LIA: eligibility (window, capacity, energy) + closest window."""
    eligible = []
    for c_id, buf_bytes, c_energy, c_offset in comm_states:
        orbit_clock = (t + c_offset) % T_orbit
        if orbit_clock >= T_comp + T_idle:
            continue
        if buf_bytes + D > T_comm * R_max:
            continue
        if c_energy < (buf_bytes + D) * q:
            continue
        wait = next_comm_entry(t, c_offset, T_comp, T_idle, T_orbit) - t
        eligible.append((c_id, wait))

    if eligible:
        eligible.sort(key=lambda x: x[1])
        return "forward", eligible[0][0]
    if e_p >= D * q:
        return "direct", None
    return "missed", None


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

def simulate(X, Y, l_star, W, D, I_total, cfg, strategy,
                seed=42, energy_trace=None):
    random.seed(seed)
    T_comp = cfg["T_comp"]
    T_idle = cfg["T_idle"]
    T_comm = cfg["T_comm"]
    R_max = cfg["R_max"]
    p_cost = cfg["p"]
    q = cfg["q"]
    delta_t = cfg["delta_t"]
    E_processor = cfg["E_processor"]
    E_comm = cfg["E_comm"]
    T_orbit = T_comp + T_idle + T_comm

    sats, _ = build_satellites(X, Y, cfg)
    if not sats:
        return SimMetrics(0, 0, 0, 0, 0, 0, 0, 0)

    processors = [s for s in sats if s["role"] == "processor"]
    communicators = [s for s in sats if s["role"] == "communicator"]
    comm_by_id = {c["id"]: c for c in communicators}
    comm_ids = [c["id"] for c in communicators]
    proc_index = {p["id"]: i for i, p in enumerate(processors)}

    delivered = []
    missed = 0

    prev_comm_bufs = {c["id"]: 0 for c in communicators}
    prev_comm_energy = {c["id"]: c["energy"] for c in communicators}

    n_steps = int(T_comp / delta_t)
    tasks_generated = 0

    for step_idx in range(n_steps):
        t = step_idx * delta_t

        if energy_trace is not None and processors:
            energy_trace.append((t, processors[0]["energy"]))

        curr_comm_bufs = {
            c["id"]: sum(sz for sz, _, _ in c["buffer"])
            for c in communicators
        }

        for p in processors:
            if tasks_generated >= I_total:
                break

            orbit_clock_p = (t + p["offset"]) % T_orbit
            if orbit_clock_p >= T_comp:
                continue

            inference_cost = W * p_cost
            if p["energy"] < inference_cost:
                missed += 1
                tasks_generated += 1
                continue

            p["energy"] -= inference_cost
            tasks_generated += 1

            # --- Routing decision ---
            if strategy in ("DirectOnly", "Homogeneous") or Y == 0:
                action, target_id = "direct", None
            else:
                comm_states = [
                    (c["id"], prev_comm_bufs[c["id"]],
                     prev_comm_energy[c["id"]], c["offset"])
                    for c in communicators
                ]
                if strategy == "Static":
                    action, target_id = _decide_static(
                        proc_index[p["id"]], comm_ids)
                elif strategy == "GreedyEVF":
                    action, target_id = _decide_greedy_evf(
                        comm_states, t, T_comp, T_idle, T_orbit)
                elif strategy == "LIA":
                    action, target_id = _decide_lia(
                        comm_states, t, p["offset"], D, R_max, q, p["energy"],
                        T_comp, T_idle, T_comm, T_orbit)
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

            # --- Execute action ---
            if action == "missed":
                missed += 1
                continue

            if action == "direct":
                tx_energy = D * q
                if p["energy"] >= tx_energy:
                    p["energy"] -= tx_energy
                    p["buffer"].append((D, t, p["id"]))
                else:
                    missed += 1
            else:
                comm_by_id[target_id]["buffer"].append((D, t, p["id"]))

        prev_comm_bufs = curr_comm_bufs
        prev_comm_energy = {c["id"]: c["energy"] for c in communicators}

    # Phase 2: drain buffers to ground
    for s in sats:
        if not s["buffer"]:
            continue
        t_entry_base = (T_comp + T_idle - s["offset"]) % T_orbit
        accumulated = 0.0
        remaining = 0
        for size, capture_time, source in s["buffer"]:
            tx_time = size / R_max
            t_entry = t_entry_base
            if t_entry <= capture_time:
                t_entry += T_orbit
            if accumulated + tx_time > T_comm:
                remaining += 1
                continue
            energy_cost = size * q
            if s["energy"] < energy_cost:
                remaining += 1
                continue
            s["energy"] -= energy_cost
            delivered.append((capture_time, t_entry + accumulated + tx_time, source))
            accumulated += tx_time
        missed += remaining

    # --- Compute metrics ---
    # Missed images are penalized with T_orbit latency (must wait for next orbit)
    proc_e = [E_processor - p["energy"] for p in processors]
    comm_e = [E_comm - c["energy"] for c in communicators]
    mp = sum(proc_e) / len(proc_e) if proc_e else 0
    mc = sum(comm_e) / len(comm_e) if comm_e else 0
    te = sum(proc_e) + sum(comm_e)

    latencies = sorted(
        [a - c for c, a, _ in delivered]
        + [T_orbit] * missed
    )
    n = len(latencies)

    if n == 0:
        return SimMetrics(0, 0, 0, 0, mp, mc, te, 0)

    return SimMetrics(
        mean_latency=sum(latencies) / n,
        median_latency=(latencies[n // 2] if n % 2
                        else (latencies[n // 2 - 1] + latencies[n // 2]) / 2),
        max_latency=latencies[-1],
        p95_latency=latencies[min(int(math.ceil(0.95 * n)) - 1, n - 1)],
        mean_proc_energy=mp,
        mean_comm_energy=mc,
        total_energy=te,
        success_rate=len(delivered) / n if n else 0,
    )

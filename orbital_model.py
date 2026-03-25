import math
import random
from dataclasses import dataclass


@dataclass
class SimMetrics:
    mean_latency: float
    median_latency: float
    max_latency: float
    p95_latency: float
    mean_proc_energy: float
    mean_comm_energy: float
    total_energy: float
    success_rate: float


def build_satellites(X, Y, cfg):
    N_active = X + Y
    T_comp = cfg["T_comp"]
    T_idle = cfg["T_idle"]
    T_comm = cfg["T_comm"]
    T_orbit = T_comp + T_idle + T_comm
    E_processor = cfg["E_processor"]
    E_comm = cfg["E_comm"]

    if N_active == 0:
        return [], T_orbit

    majority_role = "processor" if X >= Y else "communicator"
    minority_role = "communicator" if X >= Y else "processor"
    n_majority = max(X, Y)
    n_minority = min(X, Y)

    roles = [majority_role] * N_active
    if n_minority > 0:
        step = N_active / n_minority
        for k in range(n_minority):
            idx = int(round(k * step + step / 2))
            if idx >= N_active:
                idx = N_active - 1
            roles[idx] = minority_role

    sats = []
    for i in range(N_active):
        offset = i * T_orbit / N_active
        role = roles[i]
        energy = E_processor if role == "processor" else E_comm
        sats.append({
            "id": i,
            "role": role,
            "offset": offset,
            "energy": energy,
            "buffer": [],
        })

    return sats, T_orbit


def next_comm_entry(t, offset, T_comp, T_idle, T_orbit):
    """Absolute time of the next comm window entry at or after time t."""
    t_entry_base = (T_comp + T_idle - offset) % T_orbit
    if t_entry_base <= t:
        return t_entry_base + T_orbit
    return t_entry_base


def lia_decide(communicators, t, proc_offset, D, R_max, q, e_p,
               T_comp, T_idle, T_comm, T_orbit, log=None, p_id=None):
    """
    communicators: list of (c_id, buf_bytes, c_energy, c_offset)
    Returns: ("forward", c_id) | ("direct", None) | ("missed", None)
    """
    eligible = []
    for c_id, buf_bytes, c_energy, c_offset in communicators:
        orbit_clock_c = (t + c_offset) % T_orbit
        if orbit_clock_c >= T_comp + T_idle:
            if log is not None:
                log.append(f"    C{c_id}: SKIP (in comm window, clock={orbit_clock_c:.1f})")
            continue

        if buf_bytes + D > T_comm * R_max:
            if log is not None:
                log.append(f"    C{c_id}: SKIP (capacity: buf+D={buf_bytes+D} > {T_comm*R_max:.0f})")
            continue

        energy_needed = (buf_bytes + D) * q
        if c_energy < energy_needed:
            if log is not None:
                log.append(f"    C{c_id}: SKIP (energy: need={energy_needed:.4f}, have={c_energy:.4f})")
            continue

        t_comm_start = next_comm_entry(t, c_offset, T_comp, T_idle, T_orbit)
        wait_time = t_comm_start - t
        eligible.append((c_id, wait_time))

    if eligible:
        eligible.sort(key=lambda x: x[1])
        if log is not None:
            log.append(f"    Eligible: {[(cid, f'{w:.0f}s') for cid, w in eligible]}")
        return "forward", eligible[0][0]

    if e_p >= D * q:
        return "direct", None
    else:
        return "missed", None


def simulate(X, Y, l_star, W, D, I_total, cfg, strategy, seed=42, log=None, energy_trace=None):
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

    delivered = []
    missed = 0

    prev_comm_bufs = {c["id"]: 0 for c in communicators}
    prev_comm_energy = {c["id"]: c["energy"] for c in communicators}

    n_steps = int(T_comp / delta_t)
    tasks_generated = 0

    if log is not None:
        log.append(f"=== SIMULATION: strategy={strategy}, X={X}, Y={Y}, l*={l_star} ===")
        log.append(f"    W={W}, D={D}, I_total={I_total}")
        log.append(f"    T_orbit={T_orbit}, T_comp={T_comp}, T_idle={T_idle}, T_comm={T_comm}")
        log.append(f"    R_max={R_max}, p={p_cost}, q={q}, delta_t={delta_t}")
        log.append(f"    E_processor={E_processor}, E_comm={E_comm}")
        log.append(f"    Processors: {[(p['id'], round(p['offset'],1)) for p in processors]}")
        log.append(f"    Communicators: {[(c['id'], round(c['offset'],1)) for c in communicators]}")
        log.append("")

    for step_idx in range(n_steps):
        t = step_idx * delta_t

        if energy_trace is not None and processors:
            energy_trace.append((t, processors[0]["energy"]))

        curr_comm_bufs = {c["id"]: sum(sz for sz, _, _ in c["buffer"]) for c in communicators}

        for p in processors:
            if tasks_generated >= I_total:
                break

            orbit_clock_p = (t + p["offset"]) % T_orbit
            if orbit_clock_p >= T_comp:
                if log is not None:
                    log.append(f"  t={t:.0f} P{p['id']}: NOT in AOI (orbit_clock={orbit_clock_p:.1f})")
                continue

            inference_cost = W * p_cost
            if p["energy"] < inference_cost:
                missed += 1
                tasks_generated += 1
                if log is not None:
                    log.append(f"  t={t:.0f} P{p['id']}: NO ENERGY for inference (need={inference_cost:.6f}, have={p['energy']:.6f})")
                continue

            p["energy"] -= inference_cost
            tasks_generated += 1

            if strategy == "DirectOnly" or Y == 0:
                tx_energy = D * q
                if p["energy"] >= tx_energy:
                    p["energy"] -= tx_energy
                    p["buffer"].append((D, t, p["id"]))
                    if log is not None:
                        t_entry_p = next_comm_entry(t, p["offset"], T_comp, T_idle, T_orbit)
                        log.append(f"  t={t:.0f} P{p['id']}: DIRECT (next_comm={t_entry_p:.1f}, energy_left={p['energy']:.4f})")
                else:
                    missed += 1
                    if log is not None:
                        log.append(f"  t={t:.0f} P{p['id']}: MISSED (no energy for tx, need={tx_energy:.6f}, have={p['energy']:.6f})")
                continue

            comm_states = []
            for c in communicators:
                comm_states.append((c["id"], prev_comm_bufs[c["id"]],
                                    prev_comm_energy[c["id"]], c["offset"]))

            if strategy == "LIA":
                action, target_id = lia_decide(
                    comm_states, t, p["offset"], D, R_max, q, p["energy"],
                    T_comp, T_idle, T_comm, T_orbit,
                    log=log, p_id=p["id"])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            if action == "missed":
                missed += 1
                if log is not None:
                    log.append(f"  t={t:.0f} P{p['id']}: MISSED (no eligible communicator, no energy for direct)")
                continue

            if action == "direct":
                tx_energy = D * q
                p["energy"] -= tx_energy  
                p["buffer"].append((D, t, p["id"]))
                if log is not None:
                    t_entry_p = next_comm_entry(t, p["offset"], T_comp, T_idle, T_orbit)
                    log.append(f"  t={t:.0f} P{p['id']}: DIRECT (next_comm={t_entry_p:.1f}, energy_left={p['energy']:.4f})")
            else:
                comm_by_id[target_id]["buffer"].append((D, t, p["id"]))
                if log is not None:
                    t_entry_c = next_comm_entry(t, comm_by_id[target_id]["offset"], T_comp, T_idle, T_orbit)
                    t_entry_p = next_comm_entry(t, p["offset"], T_comp, T_idle, T_orbit)
                    buf = prev_comm_bufs[target_id]
                    log.append(f"  t={t:.0f} P{p['id']}: FORWARD to C{target_id} (c_next_comm={t_entry_c:.1f}, p_next_comm={t_entry_p:.1f}, c_buf={buf})")

        prev_comm_bufs = curr_comm_bufs
        prev_comm_energy = {c["id"]: c["energy"] for c in communicators}

    if log is not None:
        log.append("")
        log.append("--- PHASE 2: COMMUNICATION ---")

    for s in sats:
        if not s["buffer"]:
            continue

        t_entry_base = (T_comp + T_idle - s["offset"]) % T_orbit

        if log is not None:
            log.append(f"  {s['role'][0].upper()}{s['id']}: offset={s['offset']:.1f}, t_entry_base={t_entry_base:.1f}, buffer_pkts={len(s['buffer'])}, energy={s['energy']:.4f}")

        accumulated_time = 0.0
        remaining = []
        for packet in s["buffer"]:
            size, capture_time, source = packet
            tx_time = size / R_max

            t_entry = t_entry_base
            if t_entry <= capture_time:
                t_entry += T_orbit

            if accumulated_time + tx_time > T_comm:
                remaining.append(packet)
                if log is not None:
                    log.append(f"    OVERFLOW: pkt t={capture_time:.0f}, no time (accum={accumulated_time:.2f}/{T_comm})")
                continue

            energy_cost = size * q
            if s["energy"] < energy_cost:
                remaining.append(packet)
                if log is not None:
                    log.append(f"    NO ENERGY: pkt t={capture_time:.0f}, need={energy_cost:.4f}, have={s['energy']:.4f}")
                continue

            s["energy"] -= energy_cost
            ground_arrival = t_entry + accumulated_time + tx_time
            delivered.append((capture_time, ground_arrival, source))
            accumulated_time += tx_time
            if log is not None:
                latency = ground_arrival - capture_time
                log.append(f"    DELIVERED: t={capture_time:.0f}, t_entry={t_entry:.1f}, arrival={ground_arrival:.1f}, latency={latency:.1f}")

        missed += len(remaining)

    if log is not None:
        log.append("")
        log.append(f"--- SUMMARY: delivered={len(delivered)}, missed={missed}, total={len(delivered)+missed} ---")

    latencies = sorted(
        [arrival - capture for capture, arrival, _ in delivered]
        + [T_orbit] * missed
    )
    n = len(latencies)

    proc_energy_used = [E_processor - p["energy"] for p in processors]
    comm_energy_used = [E_comm - c["energy"] for c in communicators]
    mean_proc = sum(proc_energy_used) / len(proc_energy_used) if proc_energy_used else 0
    mean_comm = sum(comm_energy_used) / len(comm_energy_used) if comm_energy_used else 0
    total_energy = sum(proc_energy_used) + sum(comm_energy_used)

    if n == 0:
        return SimMetrics(0, 0, 0, 0, mean_proc, mean_comm, total_energy, 0)

    mean_lat = sum(latencies) / n
    median_lat = latencies[n // 2] if n % 2 == 1 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2
    max_lat = latencies[-1]
    p95_idx = min(int(math.ceil(0.95 * n)) - 1, n - 1)
    p95_lat = latencies[p95_idx]

    success_rate = len(delivered) / n if n > 0 else 0

    return SimMetrics(
        mean_latency=mean_lat,
        median_latency=median_lat,
        max_latency=max_lat,
        p95_latency=p95_lat,
        mean_proc_energy=mean_proc,
        mean_comm_energy=mean_comm,
        total_energy=total_energy,
        success_rate=success_rate,
    )

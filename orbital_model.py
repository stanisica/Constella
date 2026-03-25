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

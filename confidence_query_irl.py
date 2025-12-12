from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt

Pos = Tuple[int, int]  # (row, col)
ACTIONS = ["U", "D", "L", "R"]


# -------------------------
# Gridworld utilities
# -------------------------
def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def step(pos: Pos, action: str, size: int) -> Pos:
    r, c = pos
    if action == "U":
        r = max(0, r - 1)
    elif action == "D":
        r = min(size - 1, r + 1)
    elif action == "L":
        c = max(0, c - 1)
    elif action == "R":
        c = min(size - 1, c + 1)
    else:
        raise ValueError(f"Unknown action: {action}")
    return (r, c)


def action_utility_distance_drop(s: Pos, a: str, goal: Pos, size: int) -> float:
    """
    Proxy Q(s,a;goal): how much the action reduces Manhattan distance to goal.
    """
    d0 = manhattan(s, goal)
    s2 = step(s, a, size)
    d1 = manhattan(s2, goal)
    return float(d0 - d1)


def softmax_log_probs(utilities: List[float], beta: float) -> List[float]:
    """
    Stable log-softmax(beta*u).
    """
    scaled = [beta * u for u in utilities]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    logz = math.log(z)
    return [(x - m) - logz for x in scaled]


def sample_boltzmann_action(s: Pos, goal: Pos, size: int, beta: float, eps: float, rng: random.Random) -> str:
    """
    Human policy: with prob eps choose random action, otherwise Boltzmann on distance-drop utility.
    """
    if rng.random() < eps:
        return rng.choice(ACTIONS)

    utils = [action_utility_distance_drop(s, a, goal, size) for a in ACTIONS]
    logps = softmax_log_probs(utils, beta)
    ps = [math.exp(lp) for lp in logps]
    # sample from ps
    r = rng.random()
    cum = 0.0
    for a, p in zip(ACTIONS, ps):
        cum += p
        if r <= cum:
            return a
    return ACTIONS[-1]


# -------------------------
# Belief update + confidence
# -------------------------
def normalize(dist: Dict[Pos, float]) -> Dict[Pos, float]:
    z = sum(dist.values())
    if z <= 0:
        n = len(dist)
        return {k: 1.0 / n for k in dist}
    return {k: v / z for k, v in dist.items()}


def human_action_log_likelihood(s: Pos, a_obs: str, goal: Pos, size: int, beta: float) -> float:
    utils = [action_utility_distance_drop(s, a, goal, size) for a in ACTIONS]
    logps = softmax_log_probs(utils, beta)
    return logps[ACTIONS.index(a_obs)]


def belief_update(belief: Dict[Pos, float], s: Pos, a_obs: str, goals: List[Pos], size: int, beta: float) -> Dict[Pos, float]:
    """
    posterior(g) ∝ prior(g) * P(a_obs | s, g)
    """
    log_post: Dict[Pos, float] = {}
    for g in goals:
        prior = max(belief.get(g, 0.0), 1e-12)
        ll = human_action_log_likelihood(s, a_obs, g, size, beta)
        log_post[g] = math.log(prior) + ll

    m = max(log_post.values())
    unnorm = {g: math.exp(lp - m) for g, lp in log_post.items()}
    return normalize(unnorm)


def entropy(belief: Dict[Pos, float]) -> float:
    h = 0.0
    for p in belief.values():
        if p > 0:
            h -= p * math.log(p)
    return h


def entropy_norm(belief: Dict[Pos, float]) -> float:
    """
    Normalize entropy to [0,1] by dividing by log(|G|).
    """
    n = len(belief)
    if n <= 1:
        return 0.0
    return entropy(belief) / math.log(n)

def belief_predict_action_logprob(belief, s, a_obs, goals, size, beta):
    # log sum_g P(g) * P(a|s,g)
    # compute in log space for stability
    log_terms = []
    for g in goals:
        ll = human_action_log_likelihood(s, a_obs, g, size, beta)  # log P(a|s,g)
        log_terms.append(math.log(max(belief[g], 1e-12)) + ll)
    m = max(log_terms)
    return m + math.log(sum(math.exp(x - m) for x in log_terms))

# -------------------------
# Environment (human moves, goal switches)
# -------------------------
@dataclass
class EpisodeConfig:
    size: int = 7
    horizon: int = 50
    step_cost: float = 0.7
    goal_reward: float = 10.0
    goals: Tuple[Pos, ...] = ((0, 0), (0, 6), (6, 6))
    start_human: Pos = (6, 0)
    start_robot: Pos = (3, 3)
    t_change: int = 20  # timestep where human goal switches
    beta_human: float = 5.0
    eps_human: float = 0.1


@dataclass
class EpisodeTrace:
    regret_cum: List[float]
    queries: int
    recovery_time: Optional[int]  # steps after change until belief>0.7 on new goal; None if never


def optimal_value(s: Pos, goal: Pos, cfg: EpisodeConfig) -> float:
    """
    Best achievable return starting from state s if you go straight to goal (shortest path).
    Each step costs cfg.step_cost, reaching goal gives cfg.goal_reward.
    """
    d = manhattan(s, goal)
    # If you can reach within horizon, this is ideal.
    return cfg.goal_reward - cfg.step_cost * d


def robot_expected_Q(s: Pos, a: str, belief: Dict[Pos, float], cfg: EpisodeConfig) -> float:
    """
    Expected "Q" proxy for robot planning: expected distance reduction to likely goals.
    We also include the step cost as a mild preference (same for all actions, so optional).
    """
    exp_u = 0.0
    for g, p in belief.items():
        exp_u += p * action_utility_distance_drop(s, a, g, cfg.size)
    return exp_u


def robot_choose_action(s: Pos, belief: Dict[Pos, float], cfg: EpisodeConfig) -> str:
    """
    Greedy action that maximizes expected distance reduction.
    """
    best_a = None
    best_val = -1e9
    for a in ACTIONS:
        v = robot_expected_Q(s, a, belief, cfg)
        if v > best_val:
            best_val = v
            best_a = a
    return best_a if best_a is not None else "U"


# -------------------------
# Agents (query policies)
# -------------------------
@dataclass
class AgentConfig:
    kind: str  # "passive", "periodic", "confidence"
    query_period: int = 8          # for periodic
    entropy_threshold: float = 0.7 # normalized entropy threshold for confidence-triggered
    nll_threshold: float = 1.3



def run_episode(cfg: EpisodeConfig, agent_cfg: AgentConfig, seed: int) -> EpisodeTrace:
    rng = random.Random(seed)

    goals = list(cfg.goals)
    # Choose initial and new goals (different)
    g0 = rng.choice(goals)
    g1 = rng.choice([g for g in goals if g != g0])

    human_pos = cfg.start_human
    robot_pos = cfg.start_robot

    # belief over goals (robot's belief about human goal)
    belief = {g: 1.0 / len(goals) for g in goals}

    regret_cum = []
    cum_regret = 0.0
    queries = 0
    recovery_time = None
    recovered_at = None

    # Track when change happened and what's the new true goal
    true_goal = g0

    for t in range(cfg.horizon):
        if t == cfg.t_change:
            true_goal = g1  # human switches goal

        # --- Human acts ---
        a_h = sample_boltzmann_action(human_pos, true_goal, cfg.size, cfg.beta_human, cfg.eps_human, rng)
        human_next = step(human_pos, a_h, cfg.size)

        # --- Robot observes human action and updates belief ---
        logp = belief_predict_action_logprob(belief, human_pos, a_h, goals, cfg.size, cfg.beta_human)
        nll = -logp
        
        belief = belief_update(belief, human_pos, a_h, goals, cfg.size, cfg.beta_human)

        # Recovery detection after change: belief mass > 0.7 on new goal
        if t >= cfg.t_change and recovered_at is None:
            if belief.get(g1, 0.0) >= 0.7:
                recovered_at = t
                recovery_time = t - cfg.t_change

        # --- Query decision ---
        do_query = False
        if agent_cfg.kind == "periodic":
            if (t % agent_cfg.query_period) == 0:
                do_query = True
        elif agent_cfg.kind == "confidence":
            if entropy_norm(belief) > agent_cfg.entropy_threshold or nll > agent_cfg.nll_threshold:
                do_query = True
            else:
                do_query = False
        elif agent_cfg.kind == "passive":
            do_query = False
        else:
            raise ValueError(f"Unknown agent kind: {agent_cfg.kind}")

        if do_query:
            queries += 1
            # Oracle reveals current true goal
            belief = {g: (1.0 if g == true_goal else 0.0) for g in goals}


        # --- Robot acts (assistive movement toward inferred goal) ---
        a_r = robot_choose_action(robot_pos, belief, cfg)
        robot_next = step(robot_pos, a_r, cfg.size)

        # --- Reward / regret accounting (alignment proxy) ---
        # We measure regret as difference between value of acting toward true goal
        # vs value of robot's actual state after its move (robot "helpfulness").
        # This keeps things simple but meaningful.
        V_star = optimal_value(robot_pos, true_goal, cfg)
        V_actual = optimal_value(robot_next, true_goal, cfg)
        # regret = how much value you lost by your move vs optimal trajectory preference
        step_regret = max(0.0, V_star - V_actual)

        cum_regret += step_regret
        regret_cum.append(cum_regret)

        # Update positions
        human_pos = human_next
        robot_pos = robot_next

    return EpisodeTrace(regret_cum=regret_cum, queries=queries, recovery_time=recovery_time)


# -------------------------
# Experiment runner + plots
# -------------------------
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def run_experiment():
    cfg = EpisodeConfig()

    agents = [
        ("Passive", AgentConfig(kind="passive")),
        ("Periodic(N=8)", AgentConfig(kind="periodic", query_period=8)),
        ("Confidence(H>0.7)", AgentConfig(kind="confidence", entropy_threshold=0.7)),
    ]

    seeds = list(range(30))  # bump to 50 if you have time

    # Collect
    regrets_by_agent: Dict[str, List[List[float]]] = {name: [] for name, _ in agents}
    queries_by_agent: Dict[str, List[int]] = {name: [] for name, _ in agents}
    recovery_by_agent: Dict[str, List[Optional[int]]] = {name: [] for name, _ in agents}

    for name, a_cfg in agents:
        for sd in seeds:
            tr = run_episode(cfg, a_cfg, seed=sd)
            regrets_by_agent[name].append(tr.regret_cum)
            queries_by_agent[name].append(tr.queries)
            recovery_by_agent[name].append(tr.recovery_time)

    # Plot 1: mean cumulative regret vs time
    plt.figure()
    T = cfg.horizon
    for name, _ in agents:
        # mean across seeds at each t
        mean_curve = []
        for t in range(T):
            vals_t = [curve[t] for curve in regrets_by_agent[name]]
            mean_curve.append(mean(vals_t))
        plt.plot(range(T), mean_curve, label=name)
    plt.axvline(cfg.t_change, linestyle="--")
    plt.xlabel("t")
    plt.ylabel("Mean cumulative regret")
    plt.title("Cumulative regret over time (goal changes at dashed line)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("plot_regret_curve.png", dpi=200)

    # Plot 2: queries per episode (bar)
    plt.figure()
    names = [n for n, _ in agents]
    q_means = [mean(queries_by_agent[n]) for n in names]
    plt.bar(names, q_means)
    plt.ylabel("Mean queries per episode")
    plt.title("Human query burden")
    plt.tight_layout()
    plt.savefig("plot_queries_bar.png", dpi=200)

    # Plot 3: recovery time (boxplot; ignore None)
    plt.figure()
    rec_lists = []
    for n in names:
        vals = [v for v in recovery_by_agent[n] if v is not None]
        # if empty, add a dummy large value so plotting doesn't crash (but you should inspect)
        rec_lists.append(vals if vals else [cfg.horizon - cfg.t_change])
    plt.boxplot(rec_lists, labels=names, showmeans=True)
    plt.ylabel("Steps to recover (belief(new goal) >= 0.7)")
    plt.title("Recovery time after goal change")
    plt.tight_layout()
    plt.savefig("plot_recovery_box.png", dpi=200)

    # Print summary table
    print("\n=== Summary (mean over seeds) ===")
    for n in names:
        rec_vals = [v for v in recovery_by_agent[n] if v is not None]
        rec_mean = mean(rec_vals) if rec_vals else float("nan")
        print(f"{n:18s} | queries={mean(queries_by_agent[n]):.2f} | recovery={rec_mean:.2f}")

    print("\nSaved plots:")
    print(" - plot_regret_curve.png")
    print(" - plot_queries_bar.png")
    print(" - plot_recovery_box.png")


if __name__ == "__main__":
    run_experiment()

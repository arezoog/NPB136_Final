import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

np.random.seed(42)

# ─────────────────────────────────────────────
# TASK: Probabilistic reversal learning
# 2 options: A (initially 70% reward) and B (30%)
# Reversal at trial 50
# ─────────────────────────────────────────────

N_TRIALS = 100
N_SIMS   = 200
REVERSAL = 50

def get_reward(trial, choice):
    if trial < REVERSAL:
        p_reward = [0.70, 0.30]
    else:
        p_reward = [0.30, 0.70]
    return 1 if np.random.rand() < p_reward[choice] else 0

# ─────────────────────────────────────────────
# Q-LEARNING with asymmetric learning rates
# delta_t = r_t - Q_t(a_t)
# Q_{t+1}(a) = Q_t(a) + alpha+ * max(delta,0) * k_DA
#                      + alpha- * min(delta,0) * k_5HT_punishment   [Model A]
#            OR
# Q_{t+1}(a) = Q_t(a) + alpha+ * max(delta,0) * k_DA
#            inhibition affects beta (softmax temperature)          [Model B]
# Choice: softmax with beta
# ─────────────────────────────────────────────

def run_agent(k_DA=1.0, alpha_pos=0.3, alpha_neg=0.3,
              beta=5.0, k_5HT_punish=1.0, k_5HT_inhibit=0.0,
              n_trials=N_TRIALS, n_sims=N_SIMS):
    """
    k_DA          : scales alpha_pos (dopamine gain on positive RPE)
    k_5HT_punish  : scales alpha_neg (serotonin-as-punishment-sensitivity)
    k_5HT_inhibit : ADDED to beta (serotonin-as-behavioral-inhibition → more cautious)
    """
    all_choices  = np.zeros((n_sims, n_trials))
    all_rewards  = np.zeros((n_sims, n_trials))
    all_Q        = np.zeros((n_sims, n_trials, 2))

    for sim in range(n_sims):
        Q = np.array([0.5, 0.5])
        for t in range(n_trials):
            eff_beta = beta + k_5HT_inhibit
            p = np.exp(eff_beta * Q) / np.sum(np.exp(eff_beta * Q))
            choice = np.random.choice(2, p=p)
            reward = get_reward(t, choice)
            delta  = reward - Q[choice]

            Q[choice] += (alpha_pos * k_DA  * max(delta, 0) +
                          alpha_neg * k_5HT_punish * min(delta, 0))

            all_choices[sim, t] = choice
            all_rewards[sim, t] = reward
            all_Q[sim, t]       = Q

    return all_choices, all_rewards, all_Q

# ─────────────────────────────────────────────
# CONDITIONS
# ─────────────────────────────────────────────
conditions = {
    "Baseline":              dict(k_DA=1.0, alpha_pos=0.3, alpha_neg=0.3, beta=5.0, k_5HT_punish=1.0, k_5HT_inhibit=0.0),
    "High Dopamine":         dict(k_DA=1.8, alpha_pos=0.3, alpha_neg=0.3, beta=5.0, k_5HT_punish=1.0, k_5HT_inhibit=0.0),
    "5HT: Punishment":       dict(k_DA=1.0, alpha_pos=0.3, alpha_neg=0.3, beta=5.0, k_5HT_punish=2.0, k_5HT_inhibit=0.0),
    "5HT: Inhibition":       dict(k_DA=1.0, alpha_pos=0.3, alpha_neg=0.3, beta=5.0, k_5HT_punish=1.0, k_5HT_inhibit=4.0),
    "Depression-like":       dict(k_DA=0.4, alpha_pos=0.3, alpha_neg=0.3, beta=5.0, k_5HT_punish=1.0, k_5HT_inhibit=0.0),
    "ADHD-like":             dict(k_DA=1.0, alpha_pos=0.3, alpha_neg=0.3, beta=1.5, k_5HT_punish=1.0, k_5HT_inhibit=0.0),
}

results = {}
for name, params in conditions.items():
    c, r, q = run_agent(**params)
    results[name] = {"choices": c, "rewards": r, "Q": q}

# ─────────────────────────────────────────────
# HELPER: smooth reward and % choosing optimal
# ─────────────────────────────────────────────
def smooth(x, w=10):
    return np.convolve(x, np.ones(w)/w, mode='same')

def optimal_choice(choices):
    # optimal = A before reversal, B after
    opt = np.zeros_like(choices)
    opt[:, :REVERSAL]  = (choices[:, :REVERSAL]  == 0).astype(float)
    opt[:, REVERSAL:]  = (choices[:, REVERSAL:]  == 1).astype(float)
    return opt.mean(axis=0)

def win_stay_lose_shift(choices, rewards):
    ws, ls = [], []
    for sim in range(choices.shape[0]):
        stay = (choices[sim, 1:] == choices[sim, :-1])
        won  = rewards[sim, :-1] == 1
        lost = rewards[sim, :-1] == 0
        if won.sum() > 0:  ws.append(stay[won].mean())
        if lost.sum() > 0: ls.append((~stay[lost]).mean())
    return np.mean(ws), np.mean(ls)

def switch_rate(choices):
    return (choices[:, 1:] != choices[:, :-1]).mean()

# ─────────────────────────────────────────────
# FIGURE 1: Model schematic (SVG-style with matplotlib)
# ─────────────────────────────────────────────
# (skipping schematic — will produce 3 data figures)

# ─────────────────────────────────────────────
# FIGURE 2: Learning curves — % optimal choice
# ─────────────────────────────────────────────
COLORS = {
    "Baseline":        "#555555",
    "High Dopamine":   "#E05C2A",
    "5HT: Punishment": "#2A7EE0",
    "5HT: Inhibition": "#6E2AE0",
    "Depression-like": "#C0392B",
    "ADHD-like":       "#27AE60",
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor='white')
trials = np.arange(N_TRIALS)

# Panel A: Neuromodulator manipulations
ax = axes[0]
for name in ["Baseline", "High Dopamine", "5HT: Punishment", "5HT: Inhibition"]:
    opt = optimal_choice(results[name]["choices"])
    ax.plot(trials, smooth(opt), color=COLORS[name], lw=2, label=name)
ax.axvline(REVERSAL, color='gray', ls='--', lw=1.2, alpha=0.7, label='Reversal')
ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("P(optimal choice)", fontsize=11)
ax.set_title("A. Neuromodulator Manipulations", fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0.2, 1.0)
ax.spines[['top','right']].set_visible(False)

# Panel B: Psychiatric conditions
ax = axes[1]
for name in ["Baseline", "Depression-like", "ADHD-like"]:
    opt = optimal_choice(results[name]["choices"])
    ax.plot(trials, smooth(opt), color=COLORS[name], lw=2, label=name)
ax.axvline(REVERSAL, color='gray', ls='--', lw=1.2, alpha=0.7, label='Reversal')
ax.axhline(0.5, color='gray', ls=':', lw=1, alpha=0.5)
ax.set_xlabel("Trial", fontsize=11)
ax.set_ylabel("P(optimal choice)", fontsize=11)
ax.set_title("B. Psychiatric Parameter Sets", fontsize=12, fontweight='bold')
ax.legend(fontsize=9, loc='lower right')
ax.set_ylim(0.2, 1.0)
ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig("/home/claude/fig_learning_curves.png", dpi=180, bbox_inches='tight')
plt.close()
print("Saved fig_learning_curves.png")

# ─────────────────────────────────────────────
# FIGURE 3: Behavioral phenotype summary bar chart
# ─────────────────────────────────────────────
cond_names = list(conditions.keys())
total_reward   = [results[n]["rewards"].mean()*100   for n in cond_names]
switch_rates   = [switch_rate(results[n]["choices"])*100 for n in cond_names]
ws_rates = []; ls_rates = []
for n in cond_names:
    ws, ls = win_stay_lose_shift(results[n]["choices"], results[n]["rewards"])
    ws_rates.append(ws*100); ls_rates.append(ls*100)

x = np.arange(len(cond_names))
w = 0.2
colors = [COLORS[n] for n in cond_names]

fig, ax = plt.subplots(figsize=(13, 5), facecolor='white')
bars1 = ax.bar(x - 1.5*w, total_reward,  w, label='Avg Reward (%)', color=[c+'CC' for c in colors], edgecolor='black', lw=0.5)
bars2 = ax.bar(x - 0.5*w, switch_rates,  w, label='Switch Rate (%)', color=colors, edgecolor='black', lw=0.5)
bars3 = ax.bar(x + 0.5*w, ws_rates,      w, label='Win-Stay (%)',   color=[c+'99' for c in colors], edgecolor='black', lw=0.5)
bars4 = ax.bar(x + 1.5*w, ls_rates,      w, label='Lose-Shift (%)', color=[c+'55' for c in colors], edgecolor='black', lw=0.5)

ax.set_xticks(x)
ax.set_xticklabels(cond_names, rotation=15, ha='right', fontsize=9)
ax.set_ylabel("Percentage (%)", fontsize=11)
ax.set_title("Behavioral Phenotype Summary Across Conditions", fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.spines[['top','right']].set_visible(False)
ax.set_ylim(0, 105)
plt.tight_layout()
plt.savefig("/home/claude/fig_phenotype_summary.png", dpi=180, bbox_inches='tight')
plt.close()
print("Saved fig_phenotype_summary.png")

# ─────────────────────────────────────────────
# FIGURE 4: Q-value trajectories for key conditions
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 7), facecolor='white')
axes = axes.flatten()

for i, name in enumerate(cond_names):
    ax = axes[i]
    Q_mean = results[name]["Q"].mean(axis=0)  # (n_trials, 2)
    ax.plot(trials, Q_mean[:, 0], color='#E05C2A', lw=2, label='Q(Option A)')
    ax.plot(trials, Q_mean[:, 1], color='#2A7EE0', lw=2, label='Q(Option B)')
    ax.axvline(REVERSAL, color='gray', ls='--', lw=1, alpha=0.7)
    ax.set_title(name, fontsize=10, fontweight='bold', color=COLORS[name])
    ax.set_xlabel("Trial", fontsize=9)
    ax.set_ylabel("Q-value", fontsize=9)
    ax.set_ylim(-0.1, 1.1)
    ax.spines[['top','right']].set_visible(False)
    if i == 0:
        ax.legend(fontsize=8)

fig.suptitle("Mean Q-value Trajectories by Condition", fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig("/home/claude/fig_qvalue_trajectories.png", dpi=180, bbox_inches='tight')
plt.close()
print("Saved fig_qvalue_trajectories.png")

# Print summary stats for the write-up
print("\n=== SUMMARY STATS ===")
for n in cond_names:
    ws, ls = win_stay_lose_shift(results[n]["choices"], results[n]["rewards"])
    sr = switch_rate(results[n]["choices"])
    tr = results[n]["rewards"].mean()
    print(f"{n:25s}  reward={tr:.3f}  switch={sr:.3f}  WS={ws:.3f}  LS={ls:.3f}")

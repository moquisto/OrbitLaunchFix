import numpy as np
import matplotlib.pyplot as plt

#---------------------------------------------------
# Target (unnormalised) distribution P(x) = x e^{-x}
#---------------------------------------------------
def P(x):
    """Unnormalised target distribution P(x) = x e^{-x}, x >= 0."""
    x = np.asarray(x)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    out[mask] = x[mask] * np.exp(-x[mask])
    return out

#---------------------------------------------------
# One Metropolis Markov chain, as on the slide
#---------------------------------------------------
def metropolis_chain(delta, N_total, x0, rng):
    """
    Metropolis algorithm exactly as described in the slide:

    1. x_j = x_i + d_i, with d_i uniform in [-delta, delta]
    2. w = P(x_j)/P(x_i)
       draw r uniform in [0,1]
       if w > r: accept and set x_{i+1} = x_j
       else:     reject and set x_{i+1} = x_i
    """
    x = np.empty(N_total)
    x[0] = x0
    n_accept = 0

    for i in range(N_total - 1):
        # Step 1: propose a trial point
        d_i = rng.uniform(-delta, delta)
        x_trial = x[i] + d_i

        # P(x) is defined as 0 for x <= 0, so proposals with x_trial <= 0
        # are automatically rejected via w = 0
        w = P(x_trial) / P(x[i])

        # Step 2: draw r in [0,1] and accept if w > r
        r = rng.random()
        if w > r:
            x[i+1] = x_trial
            n_accept += 1
        else:
            x[i+1] = x[i]

    acc_rate = n_accept / (N_total - 1)
    return x, acc_rate

#---------------------------------------------------
# Run many chains for a given delta and collect stats
#---------------------------------------------------
def run_for_delta(delta, N_eq, N_meas, N_runs, seed=None):
    """
    For a fixed proposal step size delta:
      * run N_runs independent Metropolis chains
      * discard N_eq initial points (burn-in)
      * use N_meas points to estimate <x>
      * compute:
          - naive standard error sqrt(<var(x)/N_meas>)
          - RMS error relative to exact answer 2
          - mean acceptance rate
    """
    rng = np.random.default_rng(seed)
    N_total = N_eq + N_meas
    exact = 2.0

    se_sq = []       # store var(x)/N_meas for each run
    rms_err_sq = []  # store (xbar - 2)^2 for each run
    acc_rates = []

    for _ in range(N_runs):
        # You can randomise x0 if you like; 1.0 is a simple positive start
        x0 = 1.0
        chain, acc = metropolis_chain(delta, N_total, x0, rng)

        # Step 5 on the slide: skip N0 initial points and average from N0 onward
        samples = chain[N_eq:]

        xbar = samples.mean()
        var_x = samples.var(ddof=1)

        se_sq.append(var_x / len(samples))      # (σ^2 / N)
        rms_err_sq.append((xbar - exact)**2)
        acc_rates.append(acc)

    SE_naive = np.sqrt(np.mean(se_sq))
    RMS_err = np.sqrt(np.mean(rms_err_sq))
    acc_rate_mean = np.mean(acc_rates)

    return SE_naive, RMS_err, acc_rate_mean

#---------------------------------------------------
# Main script: sweep delta, print numbers, and plot
#---------------------------------------------------
def main():
    N_eq   = 2000    # N0: number of points to skip (equilibration)
    N_meas = 20000   # N: number of points used for averages
    N_runs = 100     # how many independent MC runs for the RMS statistics

    # Try delta between 0.1 and 10 on a log scale
    deltas = np.array([
    0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 1.0,
    2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0
], dtype=float)


    se_vals  = []
    rms_vals = []
    acc_vals = []

    for delta in deltas:
        SE_naive, RMS_err, acc_rate = run_for_delta(delta, N_eq, N_meas, N_runs)
        se_vals.append(SE_naive)
        rms_vals.append(RMS_err)
        acc_vals.append(acc_rate)

        print(f"delta={delta:5.3f}  SE_naive={SE_naive:.4f}  "
              f"RMS_err={RMS_err:.4f}  acc_rate={acc_rate:.3f}")

    # Acceptance rate vs delta
    plt.figure()
    plt.plot(deltas, acc_vals, 'o-', color = 'blue')
    plt.xlabel(r'step size')
    plt.ylabel('acceptance rate')
    plt.grid(True, which='both')
    plt.tight_layout()

    # Error curves vs delta (log-log)
    plt.figure()
    plt.loglog(deltas, se_vals,  'o-', label=r'RMS averaged over mutiple MC runs:', color = 'red')
    plt.loglog(deltas, rms_vals, 'o-', label=r'RMS difference of the averages with the exact answer', color = 'blue')
    plt.xlabel(r'step size')
    plt.ylabel('error')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()

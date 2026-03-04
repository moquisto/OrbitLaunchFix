from relabilityanalysis import ReliabilitySuite

if __name__ == '__main__':
    s = ReliabilitySuite(output_dir='reliability_outputs/_parallel_logic_bif', save_figures=False, show_plots=False, random_seed=1337)
    s.analyze_bifurcation_2d_map(n_thrust=3, n_density=3, max_workers=1)
    serial = dict(s._summary.get('q5_bifurcation_2d', {}))
    s.analyze_bifurcation_2d_map(n_thrust=3, n_density=3, max_workers=2)
    parallel = dict(s._summary.get('q5_bifurcation_2d', {}))

    print('serial', serial)
    print('parallel', parallel)

    keys = sorted(set(serial.keys()) | set(parallel.keys()))
    ok = True
    for k in keys:
        a = serial.get(k, None)
        b = parallel.get(k, None)
        if isinstance(a, float) and isinstance(b, float):
            if (a != a and b != b):
                continue
            if abs(a - b) > 1e-12:
                ok = False
        else:
            if a != b:
                ok = False
    print('bifurcation_logic_match', ok)

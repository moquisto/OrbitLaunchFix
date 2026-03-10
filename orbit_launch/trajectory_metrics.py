import numpy as np


def spherical_altitude_m(position_vector, env_cfg):
    r = np.asarray(position_vector, dtype=float)
    return float(np.linalg.norm(r) - float(env_cfg.earth_radius_equator))


def ellipsoidal_altitude_m(position_vector, env_cfg):
    r = np.asarray(position_vector, dtype=float)
    r_sq = float(np.dot(r, r))
    r_mag = np.sqrt(max(r_sq, 1e-16))
    r_eq = float(env_cfg.earth_radius_equator)
    r_pol = r_eq * (1.0 - float(env_cfg.earth_flattening))
    rho_sq = float(r[0] ** 2 + r[1] ** 2)
    z_sq = float(r[2] ** 2)
    denom = np.sqrt((r_pol ** 2) * rho_sq + (r_eq ** 2) * z_sq + 1e-16)
    r_local = (r_eq * r_pol * r_mag) / denom
    return float(r_mag - r_local)


def target_orbit_radius_m(target_altitude_m, env_cfg):
    return float(env_cfg.earth_radius_equator + float(target_altitude_m))


def circular_target_speed_m_s(target_altitude_m, env_cfg):
    return float(np.sqrt(float(env_cfg.earth_mu) / target_orbit_radius_m(target_altitude_m, env_cfg)))

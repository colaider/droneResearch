"""
IMU noise model for the InvenSense MPU-6000, derived from its datasheet specs rather than
guessed. Three tiers of confidence, kept explicit throughout this file:

  1. SOURCED  -- taken directly from the MPU-6000 datasheet.
  2. DERIVED  -- computed from a sourced number via a stated, standard formula.
  3. ESTIMATED -- no MPU-6000-specific spec exists for this; extrapolated with a stated
     assumption. Flagged wherever used.

Genesis's IMU sensor (genesis/engine/sensors/base_sensor.py::_apply_hardware_imperfections)
applies these every simulated step as:
    measured += bias                                  # constant offset
    measured += Normal(0, noise)                       # fresh white noise each step
    running_walk += Normal(0, random_walk); measured += running_walk   # accumulating drift

Both `noise` and `random_walk` are PER-STEP standard deviations -- Genesis does not scale
them by the timestep internally, so the conversion from a continuous-time datasheet spec
has to produce an already-per-step number, done explicitly below.
"""

import math

DEG2RAD = math.pi / 180.0
G_TO_MS2 = 9.80665

# ---- SOURCED: MPU-6000 datasheet (Product Specification Revision 3.4) ----
GYRO_NOISE_DENSITY_DEG_S_RTHZ = 0.005      # "Rate Noise Spectral Density"
ACCEL_NOISE_DENSITY_UG_RTHZ = 400.0        # "Noise Power Spectral Density"
GYRO_LF_NOISE_PP_DEG_S = 0.033             # "Low Frequency Noise", 0.1-10Hz band, 10s window
GYRO_ZRO_TOL_DEG_S = 20.0                  # "Zero-Rate Output" tolerance, raw/uncalibrated (informational only)
ACCEL_OFFSET_TOL_G = 0.05                  # accel offset tolerance X/Y, raw/uncalibrated (informational only)


def compute_mpu6000_imu_noise(
    fs_hz: float = 100.0,
    calib_duration_s: float = 2.0,
    calib_odr_hz: float = 1000.0,
):
    """
    Derive the 6 noise-model parameters Genesis's `gs.sensors.IMU` expects, from the
    MPU-6000 datasheet, for a sensor read at `fs_hz` (this repo's swarm stack calls
    `imu.read()` once per `scene.step()`, i.e. fs_hz = 1/dt = 100 Hz at dt=0.01).

    White noise (DERIVED): a datasheet noise spectral density ND (units/rtHz) converts to
    a per-sample standard deviation via sigma = ND * sqrt(fs/2) -- the Nyquist-bandwidth
    scaling standard for converting a continuous-time noise-density spec into discrete-time
    sample noise. Using fs_hz = 100 (this sim's actual IMU read rate), not the chip's max
    internal ODR (8 kHz) which this simulation never samples at.

    Bias (DERIVED, with a stated assumption): flight controllers calibrate gyro bias on
    every arm, and Betaflight-class firmware does a one-time accel calibration. Modeled
    as the residual standard error left after averaging `calib_duration_s` of raw samples
    at `calib_odr_hz` during that calibration: sigma_bias = sigma_raw_sample / sqrt(N),
    N = calib_duration_s * calib_odr_hz. This is standard-error-of-the-mean statistics
    applied to the datasheet noise density, not a separate datasheet number (MPU-6000's
    datasheet only gives the raw ZRO/offset tolerance, which is the UNcalibrated figure
    and would overstate real in-flight bias by roughly two orders of magnitude).

    Random walk / bias instability (gyro: DERIVED; accel: ESTIMATED): MPU-6000's datasheet
    gives a gyro "Low Frequency Noise" spec (0.1-10Hz band, the flicker-noise band that
    physically IS bias wander) as 0.033 deg/s peak-to-peak over a 10s window. Converted to
    a random-walk diffusion coefficient K (deg/s per sqrt(s)) via the standard peak-to-peak
    ~= 6*sigma approximation for an approximately-Gaussian process, then sigma(t) = K*sqrt(t)
    solved for K at t=10s, then scaled to this sim's per-step std via K*sqrt(dt).
    MPU-6000's datasheet does not publish an equivalent accelerometer low-frequency-noise
    spec, so the accelerometer's random walk is ESTIMATED by applying the same
    LF-noise-to-white-noise-density ratio measured for the gyro to the accelerometer's
    white noise density -- an extrapolation (both are MEMS elements on the same die, so
    comparable relative flicker-noise behavior is a reasonable but unverified assumption),
    not a sourced number.

    Returns
    -------
    dict with acc_noise, gyro_noise, acc_bias, gyro_bias, acc_random_walk, gyro_random_walk,
    each a 3-tuple (isotropic across axes) in the units gs.sensors.IMU expects (m/s^2, rad/s).
    """
    dt = 1.0 / fs_hz

    # --- white noise (DERIVED) ---
    gyro_noise_deg_s = GYRO_NOISE_DENSITY_DEG_S_RTHZ * math.sqrt(fs_hz / 2.0)
    gyro_noise_rad_s = gyro_noise_deg_s * DEG2RAD

    accel_noise_density_ms2_rthz = (ACCEL_NOISE_DENSITY_UG_RTHZ * 1e-6) * G_TO_MS2
    accel_noise_ms2 = accel_noise_density_ms2_rthz * math.sqrt(fs_hz / 2.0)

    # --- bias: post-calibration residual (DERIVED, calib duration/ODR are stated assumptions) ---
    n_calib_samples = calib_duration_s * calib_odr_hz
    gyro_noise_at_calib_odr_deg_s = GYRO_NOISE_DENSITY_DEG_S_RTHZ * math.sqrt(calib_odr_hz / 2.0)
    gyro_bias_deg_s = gyro_noise_at_calib_odr_deg_s / math.sqrt(n_calib_samples)
    gyro_bias_rad_s = gyro_bias_deg_s * DEG2RAD

    accel_noise_at_calib_odr_ms2 = accel_noise_density_ms2_rthz * math.sqrt(calib_odr_hz / 2.0)
    accel_bias_ms2 = accel_noise_at_calib_odr_ms2 / math.sqrt(n_calib_samples)

    # --- random walk / bias instability ---
    gyro_lf_sigma_deg_s = GYRO_LF_NOISE_PP_DEG_S / 6.0          # peak-to-peak ~= 6*sigma
    gyro_walk_coeff_deg_s_rts = gyro_lf_sigma_deg_s / math.sqrt(10.0)  # K, from sigma(10s) = K*sqrt(10s)
    gyro_random_walk_deg_s = gyro_walk_coeff_deg_s_rts * math.sqrt(dt)  # per-step std at this sim's fs
    gyro_random_walk_rad_s = gyro_random_walk_deg_s * DEG2RAD

    # ESTIMATED: no MPU-6000 accel LF-noise spec exists; reuse the gyro's LF/white-noise
    # density ratio (in units of sqrt(Hz)) as a stand-in "effective flicker bandwidth".
    lf_to_density_ratio_sqrt_hz = gyro_lf_sigma_deg_s / GYRO_NOISE_DENSITY_DEG_S_RTHZ
    accel_lf_sigma_ms2 = accel_noise_density_ms2_rthz * lf_to_density_ratio_sqrt_hz
    accel_walk_coeff_ms2_rts = accel_lf_sigma_ms2 / math.sqrt(10.0)
    accel_random_walk_ms2 = accel_walk_coeff_ms2_rts * math.sqrt(dt)

    return {
        "gyro_noise": (gyro_noise_rad_s,) * 3,
        "acc_noise": (accel_noise_ms2,) * 3,
        "gyro_bias": (gyro_bias_rad_s,) * 3,
        "acc_bias": (accel_bias_ms2,) * 3,
        "gyro_random_walk": (gyro_random_walk_rad_s,) * 3,
        "acc_random_walk": (accel_random_walk_ms2,) * 3,
    }

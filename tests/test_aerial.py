"""Overhead flight schedule and camera geometry regressions."""
from types import SimpleNamespace
import numpy as np
from swarm.city_stereo import AerialSurveyFlight
from swarm.compVision.stereo_depth import StereoCalibration
from swarm.sensors.stereo_rig import StereoRig


def test_downward_rig_baseline_is_horizontal_and_points_at_ground():
    scene = SimpleNamespace(add_sensor=lambda options: options)
    rig = StereoRig(scene, SimpleNamespace(idx=0), StereoCalibration(baseline_m=.5))
    left, right = rig.body_from_camera
    # A level aircraft sees along -world Z; image up is aircraft forward.
    np.testing.assert_allclose(left[:3, 2], [0, 0, -1])
    np.testing.assert_allclose(-left[:3, 1], [1, 0, 0])
    relative = np.linalg.inv(left) @ right
    np.testing.assert_allclose(relative[:3, :3], np.eye(3))
    np.testing.assert_allclose(relative[:3, 3], [.5, 0, 0])
    assert left[2, 3] < 0 and right[2, 3] < 0


def test_climb_reference_is_smooth_and_stays_over_takeoff_location():
    flight = AerialSurveyFlight()
    time = np.linspace(0, flight.climb_time, 1001)
    reference = np.array([flight.setpoint(t, [8, 0, .6]) for t in time])
    np.testing.assert_allclose(reference[:, 0], 8.)
    np.testing.assert_allclose(reference[0], [8, 0, .6, 0])
    np.testing.assert_allclose(reference[-1], [8, 0, 40, 0])
    velocity = np.gradient(reference[:, 2], time)
    assert velocity.min() >= 0 and velocity.max() <= 1.501
    assert velocity[0] < .001 and velocity[-1] < .001


def test_survey_waits_for_actual_height_then_moves_forward():
    flight = AerialSurveyFlight()
    t = flight.climb_time + 10
    assert flight.setpoint(t, [8, 0, 30])[0] == 8
    assert flight.cruise_start is None  # Time alone must not start the survey.
    assert flight.setpoint(t+1, [8, 0, 39.9])[0] == 8
    assert flight.setpoint(t+3, [8, 0, 40])[0] == 8  # Settling.
    reference = flight.setpoint(t+10, [8, 0, 40])
    assert 8 < reference[0] < 24 and reference[2] == 40
    assert flight.phase == 'survey'


def test_airborne_hover_skips_climb_and_stays_over_city():
    flight = AerialSurveyFlight(altitude=45, speed=0, start_airborne=True)
    assert flight.initial_position == (8, 0, 45)
    for t in (0, 3, 100):
        np.testing.assert_allclose(flight.setpoint(t, [8, 0, 45]), [8, 0, 45, 0])

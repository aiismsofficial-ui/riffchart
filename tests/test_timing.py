import numpy as np

from riffchart.formats.song import TempoMap
from riffchart.timing.grid import GRID, MUSICAL_POSITIONS, grid_to_sec, grid_to_ticks, phase_class, snap


def test_musical_positions_cover_16ths_32nds_and_triplets():
    pos = set(MUSICAL_POSITIONS.tolist())
    assert {0, 12, 24, 36} <= pos            # 16ths
    assert {6, 18, 30, 42} <= pos            # 32nds
    assert {16, 32, 8, 40} <= pos            # 8th triplets / sextuplets
    assert 1 not in pos and 5 not in pos     # nothing finer than 1/16 or 1/12 of a beat


def test_snap_round_trip_and_nearest_position():
    tm = TempoMap(192, np.array([0, 1920]), np.array([120.0, 90.0]))
    units = np.array([0, 12, 16, 24, 48 * 7 + 36, 48 * 12 + 32])
    t = grid_to_sec(tm, units)
    assert np.array_equal(snap(tm, t), units)
    assert np.array_equal(grid_to_ticks(tm, units), units * 4)
    # 0.26 beats -> nearest musical position is 12 (0.25 beat)
    assert snap(tm, np.array([0.13]))[0] == 12
    # just before the next beat snaps forward onto it
    assert snap(tm, np.array([0.499]))[0] == GRID


def test_phase_class():
    assert list(phase_class(np.array([0, 24, 12, 16, 6, 48 + 36]))) == [0, 1, 2, 3, 4, 2]

"""Offline checks for crazybridge.trajectory (no ROS / hardware needed)."""
import numpy as np
import pytest

from crazybridge import trajectory as tj

GOAL = (0.5, 0.0, 1.0)


def _spec(shape, **kw):
    return tj.TrajectorySpec(shape=shape, goal=GOAL, **kw)


@pytest.fixture(params=tj.SHAPES)
def spec(request):
    return _spec(request.param)


def test_piece_layout_and_firmware_limit(spec):
    pieces, duration = tj.generate(spec)
    assert pieces.shape == (spec.n_pieces, tj.ROW_LEN)
    assert pieces.shape[0] <= tj.MAX_PIECES
    assert pieces.shape[0] * tj.ROW_LEN * 4 <= 4096          # firmware memory
    assert np.isclose(pieces[:, 0].sum(), duration)
    assert np.all(pieces[:, 25:33] == 0.0)                    # yaw held at 0


@pytest.mark.parametrize('n', [0, 32, -1])
def test_rejects_bad_piece_count(n):
    with pytest.raises(ValueError):
        tj.generate(_spec('figure8', n_pieces=n))


def test_rejects_unknown_shape():
    with pytest.raises(ValueError):
        tj.generate(_spec('circle'))


def test_rejects_ramp_longer_than_trajectory():
    with pytest.raises(ValueError):
        tj.generate(_spec('figure8', ramp_s=30.0))


def test_boundary_continuity_up_to_jerk(spec):
    """C3 at every piece boundary: pos, vel, acc and jerk all match."""
    pieces, _ = tj.generate(spec)
    for d, tol in ((0, 1e-9), (1, 1e-7), (2, 1e-6), (3, 1e-5)):
        # End of piece i (tau = h) vs start of piece i+1 (tau = 0), evaluated
        # exactly -- no time offset, so real motion across the boundary can't
        # masquerade as a discontinuity.
        worst = 0.0
        for i in range(len(pieces) - 1):
            end = tj.eval_pieces(pieces[i:i + 1], [pieces[i, 0]], deriv=d)
            start = tj.eval_pieces(pieces[i + 1:i + 2], [0.0], deriv=d)
            worst = max(worst, float(np.max(np.abs(end - start))))
        assert worst < tol, f'deriv {d}: worst boundary mismatch {worst:.3e}'


def test_starts_at_goal_at_rest(spec):
    pieces, _ = tj.generate(spec)
    for d in (1, 2, 3):
        assert np.allclose(tj.eval_pieces(pieces, [0.0], deriv=d), 0.0, atol=1e-6)
    assert np.allclose(tj.eval_pieces(pieces, [0.0])[0], GOAL, atol=1e-9)


def test_ends_at_rest(spec):
    pieces, duration = tj.generate(spec)
    for d in (1, 2, 3):
        assert np.allclose(tj.eval_pieces(pieces, [duration], deriv=d), 0.0, atol=1e-5)


def test_figure8_returns_to_goal():
    pieces, duration = tj.generate(_spec('figure8'))
    assert np.allclose(tj.eval_pieces(pieces, [duration])[0], GOAL, atol=1e-6)


def test_spiral_ends_on_outer_turn():
    s = _spec('spiral')
    pieces, duration = tj.generate(s)
    end = tj.eval_pieces(pieces, [duration])[0]
    assert np.isclose(np.hypot(end[0] - GOAL[0], end[1] - GOAL[1]),
                      s.spiral_radius_m, atol=1e-6)
    assert np.isclose(end[2] - GOAL[2], s.spiral_ascent_m, atol=1e-6)


def test_polynomials_match_analytic_path(spec):
    assert tj.stats(spec)['max_fit_error_m'] < 1e-3          # plan: < 1 mm


def test_float32_quantisation_is_harmless(spec):
    """The firmware stores float32 coefficients; make sure that costs < 1 mm."""
    pieces, duration = tj.generate(spec)
    q = pieces.astype(np.float32).astype(float)
    t = np.linspace(0.0, duration, 2001)
    err = np.linalg.norm(tj.eval_pieces(q, t) - tj.eval_pieces(pieces, t), axis=1)
    assert err.max() < 1e-3


@pytest.mark.parametrize('target', [0.6, 1.0, 1.4])
def test_peak_speed_matches_request(spec, target):
    s = _spec(spec.shape, peak_speed_mps=target)
    st = tj.stats(s)
    assert abs(st['peak_speed_mps'] - target) / target < 0.05   # plan: within 5 %


def test_reference_matches_pieces_and_timescale(spec):
    pieces, duration = tj.generate(spec)
    ref = tj.sample_reference(spec, dt=0.05, timescale=2.0)
    assert ref[0, 0] == 0.0 and np.isclose(ref[-1, 0], 2.0 * duration)
    fit = tj.eval_pieces(pieces, ref[:, 0] / 2.0)
    assert np.max(np.linalg.norm(fit - ref[:, 1:4], axis=1)) < 1e-3


def test_extent_is_modest(spec):
    st = tj.stats(spec)
    assert st['max_dist_from_goal_m'] < 1.5
    print(f"\n{spec.shape}: {st}")


def test_csv_roundtrip(tmp_path, spec):
    pieces, _ = tj.generate(spec)
    path = tmp_path / 'traj.csv'
    tj.to_csv(str(path), pieces)
    assert path.read_text().splitlines()[0] == tj.CSV_HEADER
    back = np.loadtxt(path, delimiter=',', skiprows=1)
    assert np.allclose(back, pieces, rtol=1e-8, atol=1e-12)

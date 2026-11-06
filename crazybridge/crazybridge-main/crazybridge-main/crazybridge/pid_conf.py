"""The OOT controller gain file (``pid.conf``), as an object.

The file is a flat list of 24 numeric values, one per line, with ``#``-prefixed
comment lines used both as section headers and to "comment out" alternative
values. The order is fixed:

    trans kp (x, y, z)          -> values[0:3]
    trans kd (x, y, z)          -> values[3:6]
    trans ki (x, y, z)          -> values[6:9]
    trans homogeneous           -> values[9:12]    (emax, mu, gamma)
    rot   kp (x, y, z)          -> values[12:15]
    rot   kd (x, y, z)          -> values[15:18]
    rot   ki (x, y, z)          -> values[18:21]
    rot   homogeneous           -> values[21:24]   (emax, mu, gamma)

:class:`PidConf` is the single source of truth for that layout: the bridge uses
it to push gains to ``ootParams`` on connect, and ``controller_test`` uses it to
record the gains a run was flown with. It is ROS-free so it can be unit tested.

    conf = PidConf.load(path)
    for name, value in conf.params():
        cf.param.set_value(name, value)
    json.dump(conf.as_dict(), f)
"""
from __future__ import annotations

import os

# (block name, ootParams suffixes) in file order. The homogeneous blocks use
# scalar param names (no _x/_y/_z), matching the firmware's ootParams group.
LAYOUT = (
    ('trans_kp', ('trans_kp_x', 'trans_kp_y', 'trans_kp_z')),
    ('trans_kd', ('trans_kd_x', 'trans_kd_y', 'trans_kd_z')),
    ('trans_ki', ('trans_ki_x', 'trans_ki_y', 'trans_ki_z')),
    ('trans_homogeneous', ('trans_emax', 'trans_mu', 'trans_gamma')),
    ('rot_kp', ('rot_kp_x', 'rot_kp_y', 'rot_kp_z')),
    ('rot_kd', ('rot_kd_x', 'rot_kd_y', 'rot_kd_z')),
    ('rot_ki', ('rot_ki_x', 'rot_ki_y', 'rot_ki_z')),
    ('rot_homogeneous', ('rot_emax', 'rot_mu', 'rot_gamma')),
)

PARAM_GROUP = 'ootParams'

# Labels for the three components of each homogeneous block.
HOMOGENEOUS_COMPONENTS = ('emax', 'mu', 'gamma')

N_VALUES = sum(len(names) for _block, names in LAYOUT)  # 24


class PidConf:
    """A parsed ``pid.conf``: the raw values plus the named gain blocks."""

    def __init__(self, values: list[float], path: str | None = None) -> None:
        if len(values) < N_VALUES:
            where = f' in {path}' if path else ''
            raise ValueError(
                f'pid.conf needs {N_VALUES} numeric values{where}, '
                f'got {len(values)}'
            )
        self.path = path
        self.values = [float(v) for v in values[:N_VALUES]]

        self.blocks: dict[str, list[float]] = {}
        i = 0
        for block, names in LAYOUT:
            self.blocks[block] = self.values[i:i + len(names)]
            i += len(names)

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: str) -> 'PidConf':
        """Parse `path`, skipping blank and ``#``-commented lines."""
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

        values: list[float] = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                values.append(float(line))
        return cls(values, path=path)

    # -- accessors ------------------------------------------------------------
    def __getitem__(self, block: str) -> list[float]:
        return self.blocks[block]

    def params(self) -> list[tuple[str, float]]:
        """``[('ootParams.trans_kp_x', 0.45), ...]`` -- ready for set_value."""
        out: list[tuple[str, float]] = []
        for block, names in LAYOUT:
            for name, value in zip(names, self.blocks[block]):
                out.append((f'{PARAM_GROUP}.{name}', value))
        return out

    def as_dict(self) -> dict[str, float]:
        """``{'trans_kp_x': 0.45, ...}`` -- flat, JSON/CSV friendly.

        Keys are the ``ootParams`` names without the group prefix, so a recorded
        run can be diffed directly against the firmware parameters.
        """
        return {name.split('.', 1)[1]: value for name, value in self.params()}

    def describe(self) -> list[str]:
        """Human-readable one-line-per-block summary, for logging."""
        lines = []
        for block, _names in LAYOUT:
            vals = self.blocks[block]
            if block.endswith('homogeneous'):
                pairs = ', '.join(
                    f'{c} = {v}' for c, v in zip(HOMOGENEOUS_COMPONENTS, vals))
                lines.append(f'  {block}: {pairs}')
            else:
                lines.append(f'  {block} = {vals}')
        return lines

"""Where the arms are mounted.  Every task uses the same table and the same three mounts.

The table is 3.2 m long (west to east) and 1.8 m wide, its top 0.75 m high.  Two
arms face each other along its length, 1.4 m apart, each 0.9 m in from its end of the
table; the third stands at the middle of the south side, 20 cm in from its edge,
square to their line, facing north.  Each base axis is 0.70 m from the table's
centre (a Franka's base reaches 10 cm behind its axis).  Behind the west and east
arms the table goes on: a task can start things far out there.  Which robot stands
on a mount is up to the task (and could change from one combination to another);
where it stands is not.

``yaw_deg`` turns the base so the arm's +x faces the table's centre.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.errors import ConfigError

TABLE_SIZE = (3.2, 1.8, 0.75)
TABLE_TOP_Z = TABLE_SIZE[2]
#: from the table's centre to each base axis
MOUNT_R = 0.70


@dataclass(frozen=True)
class Mount:
    x: float
    y: float
    yaw_deg: float


MOUNTS: dict[str, Mount] = {
    "west": Mount(-MOUNT_R, 0.0, 0.0),
    "east": Mount(MOUNT_R, 0.0, 180.0),
    "south": Mount(0.0, -MOUNT_R, 90.0),
}


def mount(name: str) -> Mount:
    try:
        return MOUNTS[name]
    except KeyError:
        raise ConfigError(f"no mount {name!r}; the mounts are {', '.join(MOUNTS)}") from None

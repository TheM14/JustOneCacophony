"""Programme trim and front-back balance taken from the OAMD trim element.

TS 103 420 splits a programme's trim data into nine configurations, selected
by a layout's surround and height speaker counts:

    cfg = kappa_f(surrounds) + 3 * kappa_h(heights)

    kappa_f(n) = 0 for n == 0, 1 for n in (2, 3), 2 otherwise
    kappa_h(n) = 0 for n == 0, 1 for n == 2, 2 otherwise

Each configuration holds one unit of calibration:

  * a level in dB for the centre, surround and height speaker groups, and
  * a front-back balance for the listener plane and for the overhead layer.

Only the layouts listed in ``SUPPORTED_LAYOUTS`` carry these semantics.

Configurations left at default contribute neither a static level nor a balance:
the automatic attenuation is already applied per object by the layout position
compensation.  Configurations declared as disabled contribute nothing either.

The balance reparameterises the front-back axis; it does not move objects off
the axis and it leaves the two walls untouched.  See ``front_back``.
"""
from __future__ import annotations

# Layouts that carry trim and front-back balance semantics, by their compact
# names: 2.0, 3.1, 4.0, 5.1, 5.1.2, 5.1.4 and 7.1.  Wider layouts leave the
# programme untrimmed.
SUPPORTED_LAYOUTS = ("20", "31", "40", "51", "512", "514", "71")

# Channel label -> trim group.  Left, right and LFE carry no trim in any
# configuration, so they are absent from the table.
GROUPS = {
    "C": "centre",
    "Ls": "surround", "Rs": "surround",
    "Lrs": "surround", "Rrs": "surround",
    "Ltf": "height", "Rtf": "height",
    "Ltr": "height", "Rtr": "height",
    "Ltm": "height", "Rtm": "height",
}

FLOOR_LABELS = ("Ls", "Rs", "Lrs", "Rrs")
HEIGHT_LABELS = ("Ltf", "Rtf", "Ltr", "Rtr", "Ltm", "Rtm")

# bal3d_y is coded in steps of 1/16, so a balance is quantised to that grid.
BALANCE_STEPS = 16
# Front-back balance keeps a 5 percent wall band at each end and leaves it unmapped.
WALL = 0.05


def _height_class(count):
    if count == 0:
        return 0
    if count == 2:
        return 1
    return 2


def _floor_class(count):
    if count == 0:
        return 0
    if count in (2, 3):
        return 1
    return 2


def layout_classes(layout):
    """Return ``(config_index, height_count, floor_count)`` for a layout."""
    channels = tuple(getattr(layout, "channels", ()))
    heights = sum(1 for c in channels if c in HEIGHT_LABELS)
    floor = sum(1 for c in channels if c in FLOOR_LABELS)
    return _floor_class(floor) + 3 * _height_class(heights), heights, floor


def _supported(layout):
    return getattr(layout, "name", None) in SUPPORTED_LAYOUTS


def _record(atmos_meta, layout):
    if not atmos_meta or not _supported(layout):
        return None
    index, _, _ = layout_classes(layout)
    configs = atmos_meta.get("configs") or ()
    if index >= len(configs):
        return None
    return configs[index]


def group_levels_db(atmos_meta, layout):
    """Trim level in dB for the centre, surround and height speaker groups.

    Only an explicit configuration contributes a static level.  An automatic
    configuration carries no static gain: the attenuation it describes is
    already applied per object by the layout position compensation, and
    multiplying it in a second time would count it twice.
    """
    record = _record(atmos_meta, layout)
    if not isinstance(record, dict) or record.get("default") or record.get("disabled"):
        return {"centre": 0.0, "surround": 0.0, "height": 0.0}
    return {
        "centre": float(record.get("centre", 0.0)),
        "surround": float(record.get("surround", 0.0)),
        "height": float(record.get("height", 0.0)),
    }


def channel_gains(atmos_meta, layout):
    """Per-channel linear gain for the layout, in channel order."""
    levels = group_levels_db(atmos_meta, layout)
    gains = []
    for label in getattr(layout, "channels", ()):
        group = GROUPS.get(label)
        db = levels[group] if group else 0.0
        gains.append(10.0 ** (db / 20.0))
    return gains


def balance_layers(atmos_meta, layout):
    """Front-back balance for the listener plane and the overhead layer.

    Both are returned as signed magnitudes in [-1, 1]; zero means no balance.
    Only an explicit configuration carries a balance.
    """
    record = _record(atmos_meta, layout)
    if not isinstance(record, dict) or record.get("default") or record.get("disabled"):
        return 0.0, 0.0
    listener = float(record.get("bal3d_y_lis", 0.0))
    overhead = float(record.get("bal3d_y_tb", 0.0))
    return listener, overhead


def _curve(balance):
    """Segment coefficients for one balance value.

    Returns ``(a0, a1, a2, a3)`` for the two halves of the axis, or ``None``
    when the balance is zero.

    Writing ``s = |n| / 18`` with ``n`` the balance quantised to the coding
    grid, a positive balance stretches the front half by ``1 + s`` and
    compresses the back half by ``1 - s``; a negative balance does the
    reverse.  Both halves anchor at the axis midpoint and at the two walls,
    so the mapped value always stays inside the wall range.
    """
    n = int(round(abs(balance) * BALANCE_STEPS))
    if n == 0:
        return None
    s = n / 18.0
    if balance > 0:
        return 1.0 + s, -WALL * s, 1.0 - s, 0.5 + 0.45 * s
    return 1.0 - s, WALL * s, 1.0 + s, 0.5 - 0.45 * s


def front_back(position, balance):
    """Reparameterise a front-back coordinate in [0, 1] by ``balance``.

    ``0`` is the front wall and ``1`` the back wall.  Positions inside the
    wall bands are returned unchanged, and so is every position when the
    balance is zero.
    """
    curve = _curve(balance)
    if curve is None:
        return position
    if position <= WALL or position >= 1.0 - WALL:
        return position
    a0, a1, a2, a3 = curve
    if position < 0.5:
        return a0 * position + a1
    return a2 * (position - 0.5) + a3


def apply_balance(position, height, listener, overhead):
    """Blend the two balance layers by height and map the coordinate.

    ``height`` runs from 0 at the listener plane to 1 at the top layer.
    """
    if not listener and not overhead:
        return position
    low = front_back(position, listener)
    high = front_back(position, overhead)
    return (1.0 - height) * low + height * high

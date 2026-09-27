"""User-tunable POI parameters, shared by the web server, the runner and build().

Standard library only: serve_map.py validates a request body with this module,
and it must not pay for geopandas to do so.

  overture_min_confidence  an Overture place is kept when its confidence is at
                           or above this value ...
  overture_top_percent     ... OR it is in the top N% of its country x category
                           by confidence.
  speed_caps               per POI type: whether a nearby POI of that type caps
                           V_safe, at what speed (km/h), and the walking time
                           (minutes, urban / rural) of the Valhalla pedestrian
                           isochrone that is its zone (poi_isochrone.py).
  sandwich_max_length_m    a segment this long or shorter whose V_safe is
                           higher than every segment touching its ends takes
                           its along-road neighbours' V_safe
                           (sandwich_segments.py).

By default only schools cap V_safe, at 30 km/h, within a 3-minute (urban) or
5-minute (rural) walk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

POI_TYPES = ("school", "hospital", "marketplace", "shop", "bus_stop")
DEFAULT_SPEED_KMH = 30
SPEED_MIN_KMH, SPEED_MAX_KMH = 5, 130
DEFAULT_ISO_MIN_URBAN, DEFAULT_ISO_MIN_RURAL = 3, 5
ISO_MIN_MIN, ISO_MIN_MAX = 1, 30
DEFAULT_SANDWICH_MAX_LENGTH_M = 250.0
SANDWICH_MAX_LENGTH_MIN_M, SANDWICH_MAX_LENGTH_MAX_M = 0.0, 2000.0


def _default_caps() -> tuple:
    return tuple((t, t == "school", DEFAULT_SPEED_KMH) for t in POI_TYPES)


@dataclass(frozen=True)
class SpeedCap:
    poi_type: str
    enabled: bool
    speed_kmh: int
    iso_min_urban: int = DEFAULT_ISO_MIN_URBAN
    iso_min_rural: int = DEFAULT_ISO_MIN_RURAL

    def iso_minutes(self, land_use: str) -> int:
        return self.iso_min_urban if land_use == "URBAN" else self.iso_min_rural


@dataclass(frozen=True)
class PoiParams:
    overture_min_confidence: float = 0.5
    overture_top_percent: float = 95.0
    speed_caps: tuple = field(default_factory=lambda: tuple(SpeedCap(*c) for c in _default_caps()))
    sandwich_max_length_m: float = DEFAULT_SANDWICH_MAX_LENGTH_M

    def cap(self, poi_type: str) -> SpeedCap:
        return next(c for c in self.speed_caps if c.poi_type == poi_type)

    def enabled_caps(self) -> list[SpeedCap]:
        return [c for c in self.speed_caps if c.enabled]

    def isochrone_caps(self) -> list[SpeedCap]:
        """The types whose isochrones a build needs: every enabled type, and
        schools always (their zone also feeds the legacy comparison columns)."""
        return [c for c in self.speed_caps if c.enabled or c.poi_type == "school"]

    def to_dict(self) -> dict:
        return {
            "overture_min_confidence": self.overture_min_confidence,
            "overture_top_percent": self.overture_top_percent,
            "speed_caps": {c.poi_type: {"enabled": c.enabled, "speed_kmh": c.speed_kmh,
                                        "iso_min_urban": c.iso_min_urban, "iso_min_rural": c.iso_min_rural}
                           for c in self.speed_caps},
            "sandwich_max_length_m": self.sandwich_max_length_m,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, d: dict | None) -> "PoiParams":
        """Validate and build; missing keys take their defaults. Raises ValueError."""
        if d is None:
            d = {}
        if not isinstance(d, dict):
            raise ValueError("poi_params must be an object")
        unknown = set(d) - {"overture_min_confidence", "overture_top_percent", "speed_caps",
                            "sandwich_max_length_m"}
        if unknown:
            raise ValueError(f"unknown poi_params keys: {sorted(unknown)}")
        base = cls()
        min_conf = _number(d.get("overture_min_confidence", base.overture_min_confidence),
                           "overture_min_confidence", 0.0, 1.0)
        top_pct = _number(d.get("overture_top_percent", base.overture_top_percent),
                          "overture_top_percent", 0.0, 100.0)
        caps_in = d.get("speed_caps")
        if caps_in is None:
            caps_in = {}
        if not isinstance(caps_in, dict):
            raise ValueError("speed_caps must be an object keyed by POI type")
        bad = set(caps_in) - set(POI_TYPES)
        if bad:
            raise ValueError(f"unknown POI types in speed_caps: {sorted(bad)}")
        caps = []
        for c in base.speed_caps:
            given = caps_in.get(c.poi_type, {}) or {}
            if not isinstance(given, dict):
                raise ValueError(f"speed_caps.{c.poi_type} must be an object")
            enabled = given.get("enabled", c.enabled)
            if not isinstance(enabled, bool):
                raise ValueError(f"speed_caps.{c.poi_type}.enabled must be true or false")
            speed = _number(given.get("speed_kmh", c.speed_kmh), f"speed_caps.{c.poi_type}.speed_kmh",
                            SPEED_MIN_KMH, SPEED_MAX_KMH)
            if speed != int(speed):
                raise ValueError(f"speed_caps.{c.poi_type}.speed_kmh must be a whole number")
            minutes = []
            for key in ("iso_min_urban", "iso_min_rural"):
                m = _number(given.get(key, getattr(c, key)), f"speed_caps.{c.poi_type}.{key}",
                            ISO_MIN_MIN, ISO_MIN_MAX)
                if m != int(m):
                    raise ValueError(f"speed_caps.{c.poi_type}.{key} must be a whole number")
                minutes.append(int(m))
            caps.append(SpeedCap(c.poi_type, enabled, int(speed), *minutes))
        sandwich = _number(d.get("sandwich_max_length_m", base.sandwich_max_length_m),
                           "sandwich_max_length_m", SANDWICH_MAX_LENGTH_MIN_M, SANDWICH_MAX_LENGTH_MAX_M)
        return cls(float(min_conf), float(top_pct), tuple(caps), float(sandwich))

    @classmethod
    def from_json(cls, s: str | None) -> "PoiParams":
        return cls.from_dict(json.loads(s) if s else {})


def _number(value, name: str, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


DEFAULT = PoiParams()

"""Validated, JSON-serializable data contracts; UTC instants and local time stay distinct."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from zoneinfo import ZoneInfo


def instant(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamps require an explicit UTC offset")
    return dt.astimezone(timezone.utc)


def number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} out of bounds")


def whole(value, name, low=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{name} must be an integer >= {low}")


@dataclass(frozen=True)
class Show:
    work_id: str
    production_id: str
    title: str
    tags: tuple[str, ...]
    language: str = "en"
    min_age: int | None = None
    popularity: float = 0.0
    source: str = "user-supplied"

    def __post_init__(self):
        if not all(isinstance(x, str) and x for x in (self.work_id, self.production_id, self.title, self.language, self.source)):
            raise ValueError("show identifiers, title, language and source are required")
        if not isinstance(self.tags, (list, tuple)) or any(not isinstance(t, str) or not t for t in self.tags):
            raise ValueError("tags must be nonempty strings")
        object.__setattr__(self, "tags", tuple(self.tags))
        if self.min_age is not None:
            whole(self.min_age, "min_age")
        number(self.popularity, "popularity", 0, 1)


@dataclass(frozen=True)
class Performance:
    performance_id: str
    production_id: str
    city: str
    venue: str
    starts_at: str
    timezone: str
    known_at: str
    price: float | None = None  # per ticket, including fees; not an average historical price
    currency: str = "USD"
    available_seats: int | None = None
    consecutive_seats: int | None = None
    verified_at: str | None = None
    expires_at: str | None = None
    status: str = "unknown"  # available, sold_out, cancelled, unknown
    accessible: bool | None = None
    source: str = "user-supplied"

    def __post_init__(self):
        if not all(isinstance(v, str) and v for v in (self.performance_id, self.production_id, self.city, self.venue, self.currency, self.source)):
            raise ValueError("performance identifiers and provenance are required")
        start = instant(self.starts_at)
        zone = ZoneInfo(self.timezone)
        original = datetime.fromisoformat(self.starts_at.replace("Z", "+00:00"))
        # Reject nonexistent wall time or a UTC offset inconsistent with venue timezone.
        if original.replace(tzinfo=None) != start.astimezone(zone).replace(tzinfo=None) or original.utcoffset() != start.astimezone(zone).utcoffset():
            raise ValueError("starts_at wall time/offset must agree with venue timezone")
        if instant(self.known_at) > start:
            raise ValueError("performance cannot first become known after it starts")
        if self.status not in {"available", "sold_out", "cancelled", "unknown"}:
            raise ValueError("invalid performance status")
        if self.price is not None:
            number(self.price, "price", 0)
        for name in ("available_seats", "consecutive_seats"):
            if getattr(self, name) is not None:
                whole(getattr(self, name), name)
        if self.available_seats is not None and self.consecutive_seats is not None and self.consecutive_seats > self.available_seats:
            raise ValueError("consecutive_seats exceeds available_seats")
        if self.accessible is not None and not isinstance(self.accessible, bool):
            raise ValueError("accessible must be bool or null")
        if bool(self.verified_at) != bool(self.expires_at):
            raise ValueError("ticket snapshots need both verified_at and expires_at")
        if self.verified_at and not instant(self.known_at) <= instant(self.verified_at) < instant(self.expires_at):
            raise ValueError("invalid ticket snapshot interval")


@dataclass(frozen=True)
class Request:
    request_id: str
    profile_id: str
    at: str
    cities: tuple[str, ...]
    start_date: str  # inclusive venue-local date
    end_date: str
    budget: float
    party_size: int = 1
    budget_mode: str = "per_person"
    currency: str = "USD"
    preferences: dict[str, float] = field(default_factory=dict)
    member_preferences: tuple[dict[str, float], ...] = ()
    avoid_tags: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en",)
    youngest_age: int | None = None
    preferred_hour: float = 19
    requires_accessibility: bool = False
    require_consecutive: bool = False
    allow_unverified: bool = False
    seen_work_ids: tuple[str, ...] = ()
    allow_repeats: bool = True

    def __post_init__(self):
        from datetime import date
        instant(self.at)
        if not self.request_id or not self.profile_id or not self.cities:
            raise ValueError("request/profile and at least one city are required")
        if date.fromisoformat(self.start_date) > date.fromisoformat(self.end_date):
            raise ValueError("reversed local date window")
        number(self.budget, "budget", 0)
        number(self.preferred_hour, "preferred_hour", 0, 24)
        whole(self.party_size, "party_size", 1)
        if self.youngest_age is not None:
            whole(self.youngest_age, "youngest_age")
        if self.budget_mode not in {"per_person", "total"}:
            raise ValueError("budget_mode must be per_person or total")
        if not isinstance(self.currency, str) or not self.currency:
            raise ValueError("currency is required")
        for name in ("requires_accessibility", "require_consecutive", "allow_unverified", "allow_repeats"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be bool")
        for name in ("cities", "avoid_tags", "languages", "seen_work_ids"):
            v = getattr(self, name)
            if not isinstance(v, (list, tuple)) or any(not isinstance(s, str) or not s for s in v):
                raise ValueError(f"{name} must contain strings")
            object.__setattr__(self, name, tuple(v))
        if len(self.member_preferences) > self.party_size:
            raise ValueError("more member preference profiles than party members")
        for prefs in (self.preferences, *self.member_preferences):
            if not isinstance(prefs, dict):
                raise ValueError("preferences must be mappings")
            for tag, weight in prefs.items():
                if not isinstance(tag, str) or not tag:
                    raise ValueError("preference tag required")
                number(weight, "preference weight", -1, 1)
        object.__setattr__(self, "member_preferences", tuple(self.member_preferences))

    @property
    def per_person_budget(self):
        return self.budget / self.party_size if self.budget_mode == "total" else self.budget


@dataclass
class Catalog:
    shows: list[Show]
    performances: list[Performance]
    snapshot_at: str
    synthetic: bool = False

    def __post_init__(self):
        instant(self.snapshot_at)
        show_ids = [s.production_id for s in self.shows]
        perf_ids = [p.performance_id for p in self.performances]
        if len(set(show_ids)) != len(show_ids) or len(set(perf_ids)) != len(perf_ids):
            raise ValueError("duplicate production or performance ID")
        if any(p.production_id not in show_ids for p in self.performances):
            raise ValueError("performance references unknown production")

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d.get("synthetic", False), bool):
            raise ValueError("synthetic must be bool")
        return cls([Show(**s) for s in d["shows"]], [Performance(**p) for p in d["performances"]], d["snapshot_at"], d.get("synthetic", False))

    def to_dict(self):
        return asdict(self)

#!/usr/bin/env python3
"""
build_hybrid_tkg.py
====================
Builds a Hybrid ICPS Trust Knowledge Graph (TKG) by loading and merging
the real Yelp Open Dataset and CASAS Smart Home dataset, then augmenting
the merged entities with trust/conflict relations for use with Trust-MPGNN.

Author : H. Mezni

DATASETS EXPECTED ON DISK
--------------------------
Yelp Open Dataset (https://www.yelp.com/dataset):
    <yelp_dir>/yelp_academic_dataset_business.json   # one JSON object per line
    <yelp_dir>/yelp_academic_dataset_user.json        # one JSON object per line
    <yelp_dir>/yelp_academic_dataset_review.json      # one JSON object per line

CASAS Smart Home dataset – this script handles BOTH formats released by WSU:
  Legacy text format (e.g. the "aruba", "hh101" testbeds):
      <casas_dir>/<home_name>          # raw whitespace-delimited event file
      e.g.  casas_dir/aruba
            casas_dir/hh101
  CSV / directory format (newer releases):
      <casas_dir>/<home_name>/event.csv     # comma-separated events
      <casas_dir>/<home_name>/dataset.json  # sensor metadata (optional)

  The script auto-detects the format for each home sub-directory found.

  The Kaggle variant (ashley6009/casas-smart-home-dataset) ships CSV files
  named after home IDs (aruba.csv, milan.csv, cairo.csv, tulum.csv) with
  columns:  date, time, sensor, message[, activity]
  Place those CSV files directly inside <casas_dir>/ and the script will
  pick them up automatically.

USAGE
-----
    python3 build_hybrid_tkg.py \\
        --yelp-dir  /path/to/yelp_dataset \\
        --casas-dir /path/to/casas_dataset \\
        --output    hybrid_tkg.json \\
        --max-businesses 2000 \\
        --resources-per-service 1 5 \\
        --conflict-density 0.20 \\
        --trust-density    0.08 \\
        --seed 42

    All dataset arguments are optional for a dry-run / testing mode
    (the script will warn about missing paths and process whatever it finds).

OUTPUT FORMAT
-------------
A single JSON file that matches the TKG schema consumed by Trust-MPGNN's
tkg/builder.py and tkg/instance_generator.py:

{
  "metadata": {
    "name": str,
    "yelp_source": str,          # path used
    "casas_source": str,         # path used
    "num_providers": int,
    "num_services": int,
    "num_resources": int,
    "num_edges": int,
    "conflict_density": float,
    "seed": int
  },
  "providers": [ {...}, ... ],   # Yelp users with high review activity
  "services":  [ {...}, ... ],   # Yelp businesses → IoT services
  "resources": [ {...}, ... ],   # CASAS sensor types → IoT resources
  "edges":     [ {...}, ... ]    # trust / conflict / support / oppose / allied
}

DESIGN NOTES
------------
1. Yelp businesses become *IoT services*:
      business_id   → service id   (prefixed "S_")
      name          → service name
      categories    → capability (first Yelp category, lower-cased)
      stars         → base QoS reliability proxy (normalised to [0,1])
      review_count  → QoS availability proxy
      is_open       → availability flag
      city/state    → region (carried into provider)

2. Yelp users become *providers* (business owners / operators):
      user_id       → provider id  (prefixed "P_")
      name          → provider name
      average_stars → reputation
      elite         → trust tier hint
      The provider-service link is derived from the review.json:
      the most-prolific reviewer for a business is assigned as its provider.

3. CASAS sensor events become *IoT resources*:
      Each unique sensor identifier seen across all loaded home files
      becomes one resource:
          sensor_id      → resource id   (prefixed "R_")
          sensor prefix  → resource type (M=sensor, D=door_sensor, T=sensor,
                           I=item_sensor, AD=actuator, P=phone, L=smart_lock,
                           BA=area_sensor, Area→area_sensor, etc.)
          home_name      → region / deployment context
          sensor states  → communication_protocol heuristic
          activity_labels seen with this sensor → privacy_regime heuristic

4. Resource assignment (1–5 resources per service):
      Resources are drawn from the CASAS pool using a deterministic but
      pseudo-random scheme seeded from the business_id so it is reproducible
      given the same input data.

5. Trust / conflict edges are generated stochastically per §6.2 of the paper:
      PROVIDER-PROVIDER TRUST  : driven by shared elite-year overlap and
                                 mutual review activity (Yelp social graph
                                 proxy; exact friend-graph is unavailable in
                                 the academic dataset).
      SERVICE-RESOURCE edges   : SUPPORT / OPPOSE / NEUTRAL / TRUST based on
                                 protocol-compatibility heuristics between the
                                 service's Yelp category and the CASAS sensor
                                 type, plus the configured conflict_density.
      SERVICE-SERVICE ALLIED   : services sharing ≥1 resource.
      RESOURCE-RESOURCE CONFLICT: same-type sensors in overlapping deployments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_hybrid_tkg")


# ---------------------------------------------------------------------------
# Schema dataclasses (match Trust-MPGNN tkg/dataset_generator.py output)
# ---------------------------------------------------------------------------

@dataclass
class Provider:
    id: str
    name: str
    region: str
    reputation: float        # [0, 1]
    yelp_user_id: str = ""
    elite_years: list = field(default_factory=list)


@dataclass
class Service:
    id: str
    name: str
    provider_id: str
    category: str
    qos: dict
    resource_ids: list = field(default_factory=list)
    yelp_business_id: str = ""
    is_open: bool = True


@dataclass
class Resource:
    id: str
    name: str
    type: str
    protocol: str
    data_format: str
    privacy_regime: str
    energy_profile_mw: float
    qos: dict
    casas_sensor_id: str = ""
    casas_home: str = ""


# ---------------------------------------------------------------------------
# CASAS: sensor-prefix → resource_type mapping (from dataset documentation)
# ---------------------------------------------------------------------------
CASAS_PREFIX_TO_TYPE: dict[str, str] = {
    "M":    "sensor",          # PIR motion detector
    "D":    "door_sensor",     # magnetic door / cabinet sensor
    "T":    "sensor",          # temperature sensor
    "I":    "actuator",        # item-use / contact sensor
    "AD":   "actuator",        # analogue (water level / burner)
    "P":    "actuator",        # phone-use sensor
    "L":    "smart_lock",      # light switch
    "LL":   "smart_lock",
    "BA":   "area_sensor",     # battery / zone sensor
    "E":    "sensor",          # electricity sensor (some testbeds)
}

# Location-name keywords → resource_type (for location-named sensors like
# "Bathroom", "BedroomArea", "OfficeChair" found in community-home testbeds)
LOCATION_TO_TYPE: dict[str, str] = {
    "motion":   "sensor",
    "door":     "door_sensor",
    "temp":     "sensor",
    "light":    "smart_lock",
    "bed":      "wearable",
    "bath":     "sensor",
    "kitchen":  "actuator",
    "stove":    "actuator",
    "chair":    "wearable",
    "phone":    "actuator",
    "cabinet":  "door_sensor",
    "area":     "area_sensor",
    "camera":   "camera",
    "micro":    "microphone",
    "display":  "display",
    "speaker":  "speaker",
    "lock":     "smart_lock",
    "hvac":     "hvac_actuator",
}

# Yelp category keyword → compatible CASAS resource types
# (used for SUPPORT/OPPOSE heuristic; §6.2 "protocol compatibility")
CATEGORY_RESOURCE_COMPAT: dict[str, set[str]] = {
    "restaurant":  {"sensor", "actuator", "door_sensor"},
    "food":        {"sensor", "actuator"},
    "health":      {"wearable", "sensor", "area_sensor"},
    "medical":     {"wearable", "sensor"},
    "fitness":     {"wearable", "sensor"},
    "hotel":       {"door_sensor", "sensor", "smart_lock"},
    "home":        {"sensor", "actuator", "door_sensor", "hvac_actuator"},
    "office":      {"area_sensor", "smart_lock", "display", "camera"},
    "education":   {"display", "camera", "microphone"},
    "shop":        {"camera", "smart_lock", "sensor"},
    "retail":      {"camera", "smart_lock", "sensor"},
    "beauty":      {"sensor", "wearable"},
    "automotive":  {"actuator", "sensor"},
    "nightlife":   {"camera", "microphone", "speaker"},
    "arts":        {"camera", "display", "speaker"},
    "parks":       {"area_sensor", "sensor"},
    "financial":   {"camera", "smart_lock", "display"},
    "technology":  {"display", "actuator", "sensor", "camera"},
}

PROTOCOL_OPTIONS = ["Zigbee", "Bluetooth 5.0", "Wi-Fi 6", "Z-Wave"]
DATA_FORMAT_OPTIONS = ["JSON", "XML", "FHIR", "HL7"]
PRIVACY_OPTIONS = ["GDPR", "HIPAA", "CCPA"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_seed(value: str, seed: int) -> int:
    """Deterministic integer from a string + global seed."""
    h = hashlib.md5(f"{seed}:{value}".encode()).hexdigest()
    return int(h, 16)


def _norm_stars(stars: float) -> float:
    """Map Yelp stars [1, 5] to [0, 1]."""
    return max(0.0, min(1.0, (stars - 1.0) / 4.0))


def _sensor_prefix(sensor_id: str) -> str:
    """Extract the alphabetic prefix from a CASAS sensor ID."""
    m = re.match(r"^([A-Za-z]+)", sensor_id)
    return m.group(1).upper() if m else "M"


def _sensor_type_from_id(sensor_id: str) -> str:
    """Derive resource_type from a CASAS sensor ID."""
    sid = sensor_id.upper()
    # Two-char prefixes first
    for prefix in ["AD", "LL", "BA"]:
        if sid.startswith(prefix):
            return CASAS_PREFIX_TO_TYPE.get(prefix, "sensor")
    # Location-named sensors (community-home format)
    sl = sensor_id.lower()
    for kw, rtype in LOCATION_TO_TYPE.items():
        if kw in sl:
            return rtype
    # Single-char prefix
    prefix = _sensor_prefix(sensor_id)
    return CASAS_PREFIX_TO_TYPE.get(prefix, "sensor")


def _category_compat(category: str, resource_type: str) -> bool:
    """True if the Yelp category is compatible with the resource type."""
    cat_lower = category.lower()
    for kw, rtypes in CATEGORY_RESOURCE_COMPAT.items():
        if kw in cat_lower:
            return resource_type in rtypes
    return True  # unknown category: assume compatible


def _best_category(raw_categories: str | None) -> str:
    """Extract and normalise the primary Yelp category string."""
    if not raw_categories:
        return "generic_service"
    parts = [p.strip().lower().replace(" ", "_") for p in raw_categories.split(",")]
    # prefer a non-trivial category
    skip = {"restaurants", "food", "shopping"}
    for p in parts:
        if p and p not in skip:
            return p
    return parts[0] if parts else "generic_service"


# ---------------------------------------------------------------------------
# CASAS loader
# ---------------------------------------------------------------------------

class CASASLoader:
    """Loads all sensor IDs and their associated activity labels from one or
    more CASAS home directories / files.  Supports both the legacy
    whitespace-delimited text format and the newer CSV format (including the
    Kaggle "ashley6009" CSV files).

    After calling load(), self.sensors is a dict:
        sensor_id -> {"home": str, "type": str, "activities": set[str],
                       "event_count": int, "states": set[str]}
    """

    # Expected column indices for both file formats
    # Legacy:  date time sensor message [activity]
    # CSV new: date,time,sensor,message[,activity]
    # Kaggle:  date,time,sensor,message[,activity]  (same)

    def __init__(self, casas_dir: str | Path):
        self.casas_dir = Path(casas_dir)
        self.sensors: dict[str, dict] = {}

    # ------------------------------------------------------------------
    def load(self) -> "CASASLoader":
        if not self.casas_dir.exists():
            log.warning(f"CASAS directory not found: {self.casas_dir} — skipping CASAS.")
            return self

        log.info(f"Loading CASAS data from: {self.casas_dir}")
        found = 0

        # Walk through all files/subdirectories
        for entry in sorted(self.casas_dir.iterdir()):
            if entry.is_file():
                # Could be a Kaggle-style CSV (aruba.csv, milan.csv …)
                # or a legacy raw file (aruba, hh101 …)
                if entry.suffix.lower() == ".csv":
                    found += self._load_csv_file(entry)
                elif entry.suffix == "" or entry.suffix.lower() == ".txt":
                    found += self._load_legacy_file(entry, home_name=entry.stem)
            elif entry.is_dir():
                # Modern directory-format testbed
                event_csv = entry / "event.csv"
                meta_json = entry / "dataset.json"
                if event_csv.exists():
                    found += self._load_csv_file(event_csv, home_name=entry.name,
                                                  meta_path=meta_json if meta_json.exists() else None)
                else:
                    # Try to find a raw file inside
                    for f in entry.iterdir():
                        if f.is_file() and f.suffix == "":
                            found += self._load_legacy_file(f, home_name=entry.name)
                            break

        log.info(f"  → CASAS: {len(self.sensors)} unique sensors from {found} events.")
        return self

    # ------------------------------------------------------------------
    def _register(self, sensor_id: str, home: str, message: str, activity: str):
        """Register one event into the sensor registry."""
        if not sensor_id or sensor_id.strip() == "":
            return
        sid = sensor_id.strip()
        if sid not in self.sensors:
            self.sensors[sid] = {
                "home": home,
                "type": _sensor_type_from_id(sid),
                "activities": set(),
                "event_count": 0,
                "states": set(),
            }
        self.sensors[sid]["event_count"] += 1
        if message:
            self.sensors[sid]["states"].add(message.strip().upper())
        if activity and activity.strip().lower() not in ("", "none", "other_activity"):
            self.sensors[sid]["activities"].add(activity.strip())

    # ------------------------------------------------------------------
    def _load_legacy_file(self, path: Path, home_name: str = "") -> int:
        """Load a CASAS legacy whitespace-delimited text file.
        Format (per pyActLearn docs):
            YYYY-MM-DD  HH:MM:SS.ffffff   SensorID   Message   [Activity]
        """
        home = home_name or path.stem
        count = 0
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    # Need at least: date time sensor message
                    if len(parts) < 4:
                        continue
                    # parts[0]=date, parts[1]=time, parts[2]=sensor, parts[3]=message
                    # parts[4+] may be activity (can be multiple words)
                    sensor_id = parts[2]
                    message = parts[3]
                    activity = " ".join(parts[4:]) if len(parts) > 4 else ""
                    self._register(sensor_id, home, message, activity)
                    count += 1
        except OSError as e:
            log.warning(f"  Could not read {path}: {e}")
        return count

    # ------------------------------------------------------------------
    def _load_csv_file(self, path: Path, home_name: str = "",
                       meta_path: Path | None = None) -> int:
        """Load a CASAS CSV event file.
        Expected columns (from pyActLearn docs and Kaggle variant):
            date, time, sensor, message[, activity[, activity_person_b]]
        """
        import csv

        home = home_name or path.stem
        count = 0

        # Optionally enrich sensor metadata from dataset.json
        sensor_meta: dict[str, dict] = {}
        if meta_path:
            try:
                with open(meta_path) as mf:
                    meta = json.load(mf)
                for s in meta.get("sensors", []):
                    if "name" in s:
                        sensor_meta[s["name"]] = s
            except Exception as e:
                log.debug(f"Could not load CASAS metadata from {meta_path}: {e}")

        try:
            with open(path, encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    # Skip header rows
                    if row[0].lower() in ("date", "datetime", "#"):
                        continue
                    # Tolerate variable column counts
                    if len(row) < 4:
                        continue
                    # date=row[0], time=row[1], sensor=row[2], message=row[3]
                    sensor_id = row[2].strip()
                    message = row[3].strip()
                    # Activity may be in col 4 or 5 (multi-resident format)
                    activity = row[4].strip() if len(row) > 4 else ""
                    if activity == "" and len(row) > 5:
                        activity = row[5].strip()
                    self._register(sensor_id, home, message, activity)
                    count += 1
        except OSError as e:
            log.warning(f"  Could not read {path}: {e}")
        return count

    # ------------------------------------------------------------------
    def unique_sensor_types(self) -> set[str]:
        return {v["type"] for v in self.sensors.values()}


# ---------------------------------------------------------------------------
# Yelp loader
# ---------------------------------------------------------------------------

class YelpLoader:
    """Loads businesses, users, and the user→business mapping from Yelp JSON
    files (one JSON object per line, as per Yelp's official distribution).

    The user-business mapping is derived from review.json: the user who wrote
    the most reviews for a given business is taken as its "provider", mirroring
    the paper's framing of Yelp users as IoT service providers with a trust
    history built from past interactions.
    """

    def __init__(self, yelp_dir: str | Path):
        self.yelp_dir = Path(yelp_dir)
        self.businesses: list[dict] = []
        self.users: dict[str, dict] = {}          # user_id -> user record
        self.business_top_user: dict[str, str] = {}  # business_id -> user_id

    # ------------------------------------------------------------------
    def _jsonl_path(self, filename: str) -> Path | None:
        """Locate a Yelp JSONL file, trying both underscore and original names."""
        for name in [filename, filename.replace("yelp_academic_dataset_", "")]:
            p = self.yelp_dir / name
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    def _iter_jsonl(self, path: Path, max_lines: int | None = None):
        """Iterate over records in a newline-delimited JSON file."""
        count = 0
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                        count += 1
                        if max_lines and count >= max_lines:
                            break
                    except json.JSONDecodeError:
                        continue
        except OSError as e:
            log.warning(f"  Could not read {path}: {e}")

    # ------------------------------------------------------------------
    def load_businesses(self, max_businesses: int = 2000,
                        open_only: bool = False) -> "YelpLoader":
        path = self._jsonl_path("yelp_academic_dataset_business.json")
        if path is None:
            log.warning("business.json not found — no Yelp businesses loaded.")
            return self
        log.info(f"Loading Yelp businesses from: {path}")
        for rec in self._iter_jsonl(path, max_lines=max_businesses * 3):
            if open_only and not rec.get("is_open", 1):
                continue
            if not rec.get("categories"):
                continue
            self.businesses.append(rec)
            if len(self.businesses) >= max_businesses:
                break
        log.info(f"  → {len(self.businesses)} businesses loaded.")
        return self

    # ------------------------------------------------------------------
    def load_users(self, max_users: int = 5000) -> "YelpLoader":
        path = self._jsonl_path("yelp_academic_dataset_user.json")
        if path is None:
            log.warning("user.json not found — no Yelp users loaded.")
            return self
        log.info(f"Loading Yelp users from: {path}")
        count = 0
        for rec in self._iter_jsonl(path, max_lines=max_users * 2):
            uid = rec.get("user_id", "")
            if uid:
                self.users[uid] = rec
                count += 1
                if count >= max_users:
                    break
        log.info(f"  → {len(self.users)} users loaded.")
        return self

    # ------------------------------------------------------------------
    def build_business_provider_map(self, max_reviews: int = 500_000) -> "YelpLoader":
        """Assign the most-prolific reviewer of each business as its provider.
        Falls back to random assignment if review.json is absent.
        """
        path = self._jsonl_path("yelp_academic_dataset_review.json")
        if path is None:
            log.warning("review.json not found — provider assignment will be random.")
            return self

        log.info(f"Building business→provider map from: {path}")
        biz_ids = {b["business_id"] for b in self.businesses}
        # user_id -> {business_id -> review_count}
        user_biz_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        count = 0
        for rec in self._iter_jsonl(path, max_lines=max_reviews):
            bid = rec.get("business_id", "")
            uid = rec.get("user_id", "")
            if bid in biz_ids and uid in self.users:
                user_biz_counts[uid][bid] += 1
                count += 1
        log.info(f"  → processed {count} reviews.")

        # For each business, pick the user with most reviews
        biz_candidates: dict[str, list[tuple[int, str]]] = defaultdict(list)
        for uid, biz_dict in user_biz_counts.items():
            for bid, cnt in biz_dict.items():
                biz_candidates[bid].append((cnt, uid))

        for bid, candidates in biz_candidates.items():
            self.business_top_user[bid] = max(candidates, key=lambda x: x[0])[1]

        log.info(f"  → {len(self.business_top_user)} businesses assigned a provider.")
        return self


# ---------------------------------------------------------------------------
# Trust / relation edge generator
# ---------------------------------------------------------------------------

class TrustEdgeGenerator:
    """Generates stochastic trust, conflict, support, oppose, allied, and
    OFFERS edges following the paper's §6.2 protocol.

    Randomness is fully seeded for reproducibility.
    """

    def __init__(self, seed: int, conflict_density: float, provider_trust_density: float):
        self.rng = random.Random(seed)
        self.conflict_density = conflict_density
        self.provider_trust_density = provider_trust_density

    # ------------------------------------------------------------------
    def provider_trust_edges(self, providers: list[Provider]) -> list[dict]:
        """PROVIDER → PROVIDER TRUST edges.

        Probability of a trust edge p_i → p_j is proportional to p_j's
        reputation (higher reputation → more likely to be trusted by peers),
        matching the paper's description of trust seeded from historical
        collaboration records and service ratings.
        """
        edges = []
        n = len(providers)
        target = int(self.provider_trust_density * n * (n - 1))
        seen: set[tuple[str, str]] = set()
        attempts = 0
        max_attempts = max(target * 20, 1000)
        while len(edges) < target and attempts < max_attempts:
            attempts += 1
            a, b = self.rng.sample(providers, 2)
            if (a.id, b.id) in seen:
                continue
            seen.add((a.id, b.id))
            if self.rng.random() < (0.3 + 0.6 * b.reputation):
                edges.append({
                    "src": a.id, "dst": b.id,
                    "src_type": "Provider", "dst_type": "Provider",
                    "relation": "TRUST",
                    "weight": round(0.5 + 0.5 * b.reputation, 3),
                })
        return edges

    # ------------------------------------------------------------------
    def service_resource_edges(self, services: list[Service],
                                resources_by_id: dict[str, Resource]) -> list[dict]:
        """SERVICE → RESOURCE SUPPORT / OPPOSE / NEUTRAL / TRUST edges.

        Relation type is derived from:
        (a) conflict_density — forces OPPOSE with probability conflict_density
        (b) category-resource compatibility heuristic
        (c) reliability proxy from QoS
        """
        edges = []
        for s in services:
            for rid in s.resource_ids:
                r = resources_by_id.get(rid)
                if r is None:
                    continue
                # (a) Conflict injection
                if self.rng.random() < self.conflict_density:
                    relation = "OPPOSE"
                    weight = round(self.rng.uniform(0.0, 0.3), 3)
                else:
                    # (b) Compatibility heuristic
                    compatible = _category_compat(s.category, r.type)
                    # (c) QoS quality tier
                    s_rel = s.qos.get("reliability", 0.5)
                    r_rel = r.qos.get("reliability", 0.5)
                    if compatible and s_rel > 0.9 and r_rel > 0.9:
                        relation = "TRUST"
                        weight = round(self.rng.uniform(0.8, 1.0), 3)
                    elif compatible and s_rel > 0.6:
                        relation = "SUPPORT"
                        weight = round(self.rng.uniform(0.6, 0.9), 3)
                    else:
                        relation = "NEUTRAL"
                        weight = round(self.rng.uniform(0.4, 0.6), 3)
                edges.append({
                    "src": s.id, "dst": rid,
                    "src_type": "Service", "dst_type": "Resource",
                    "relation": relation,
                    "weight": weight,
                })
        return edges

    # ------------------------------------------------------------------
    def allied_edges(self, services: list[Service]) -> list[dict]:
        """SERVICE → SERVICE ALLIED edges for services sharing ≥1 resource."""
        edges = []
        by_resource: dict[str, list[Service]] = defaultdict(list)
        for s in services:
            for rid in s.resource_ids:
                by_resource[rid].append(s)
        seen: set[tuple[str, str]] = set()
        for rid, svcs in by_resource.items():
            if len(svcs) < 2:
                continue
            for i in range(len(svcs)):
                for j in range(i + 1, len(svcs)):
                    a, b = svcs[i], svcs[j]
                    key = tuple(sorted((a.id, b.id)))
                    if key not in seen:
                        seen.add(key)
                        edges.append({
                            "src": a.id, "dst": b.id,
                            "src_type": "Service", "dst_type": "Service",
                            "relation": "ALLIED",
                            "weight": round(self.rng.uniform(0.6, 1.0), 3),
                        })
        return edges

    # ------------------------------------------------------------------
    def resource_conflict_edges(self, resources: list[Resource]) -> list[dict]:
        """RESOURCE → RESOURCE CONFLICT edges for same-type contention."""
        edges = []
        by_type: dict[str, list[Resource]] = defaultdict(list)
        for r in resources:
            by_type[r.type].append(r)
        for rtype, items in by_type.items():
            if len(items) < 2:
                continue
            n_pairs = int(self.conflict_density * len(items))
            for _ in range(n_pairs):
                a, b = self.rng.sample(items, 2)
                if a.id != b.id:
                    edges.append({
                        "src": a.id, "dst": b.id,
                        "src_type": "Resource", "dst_type": "Resource",
                        "relation": "CONFLICT",
                        "weight": round(self.rng.uniform(0.0, 0.4), 3),
                    })
        return edges

    # ------------------------------------------------------------------
    def offers_edges(self, services: list[Service]) -> list[dict]:
        """PROVIDER -OFFERS-> SERVICE structural edges."""
        return [
            {
                "src": s.provider_id, "dst": s.id,
                "src_type": "Provider", "dst_type": "Service",
                "relation": "OFFERS", "weight": 1.0,
            }
            for s in services
        ]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

class HybridTKGBuilder:
    def __init__(
        self,
        yelp_dir: str | Path,
        casas_dir: str | Path,
        max_businesses: int = 2000,
        min_resources_per_service: int = 1,
        max_resources_per_service: int = 5,
        conflict_density: float = 0.20,
        provider_trust_density: float = 0.08,
        max_users: int = 5000,
        seed: int = 42,
    ):
        self.yelp_dir = Path(yelp_dir) if yelp_dir else None
        self.casas_dir = Path(casas_dir) if casas_dir else None
        self.max_businesses = max_businesses
        self.min_res = min_resources_per_service
        self.max_res = max_resources_per_service
        self.conflict_density = conflict_density
        self.provider_trust_density = provider_trust_density
        self.max_users = max_users
        self.seed = seed
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    def _build_resources_from_casas(self, casas: CASASLoader) -> list[Resource]:
        """One Resource per unique CASAS sensor ID."""
        resources = []
        for sensor_id, info in casas.sensors.items():
            rtype = info["type"]
            home = info["home"]

            # Protocol: heuristic from sensor type + observed states
            states = info.get("states", set())
            if "ON" in states or "OFF" in states:
                protocol = self.rng.choice(["Zigbee", "Z-Wave"])
            elif any(isinstance(s, str) and s.replace(".", "").isdigit() for s in states):
                protocol = self.rng.choice(["Wi-Fi 6", "Bluetooth 5.0"])
            else:
                protocol = self.rng.choice(PROTOCOL_OPTIONS)

            # Data format: heuristic from activity labels
            activities = info.get("activities", set())
            if any(a.lower() in ("meal_preparation", "cook", "eat") for a in activities):
                data_format = "FHIR"   # healthcare / life-quality context
            elif any("sleep" in a.lower() or "bed" in a.lower() for a in activities):
                data_format = "HL7"
            else:
                data_format = self.rng.choice(["JSON", "XML"])

            # Privacy: heuristic from activity labels
            if any(a.lower() in ("sleep", "personal_hygiene", "bathing") for a in activities):
                privacy = "HIPAA"
            elif any("health" in a.lower() or "medical" in a.lower() for a in activities):
                privacy = "HIPAA"
            elif home.startswith("hh") or home.startswith("rw"):
                privacy = "GDPR"
            else:
                privacy = self.rng.choice(PRIVACY_OPTIONS)

            # QoS: driven by event frequency (busier sensor = higher availability)
            event_count = info.get("event_count", 1)
            availability = min(0.999, 0.80 + 0.19 * min(event_count / 1000.0, 1.0))
            resources.append(
                Resource(
                    id=f"R_{sensor_id}_{home}",
                    name=f"{sensor_id}-{home}",
                    type=rtype,
                    protocol=protocol,
                    data_format=data_format,
                    privacy_regime=privacy,
                    energy_profile_mw=round(self.rng.uniform(5.0, 4000.0), 2),
                    qos={
                        "response_time_ms": round(self.rng.uniform(10.0, 1000.0), 2),
                        "reliability": round(self.rng.uniform(0.70, 0.999), 4),
                        "cost": round(self.rng.uniform(0.01, 20.0), 2),
                        "availability": round(availability, 4),
                    },
                    casas_sensor_id=sensor_id,
                    casas_home=home,
                )
            )
        return resources

    # ------------------------------------------------------------------
    def _build_providers_from_yelp(self, yelp: YelpLoader) -> tuple[dict[str, Provider], set[str]]:
        """Build providers from Yelp users who are assigned to ≥1 business."""
        assigned_users: set[str] = set(yelp.business_top_user.values())
        # Also include users who have businesses but no review assignment
        # (fallback: include top users by review count)
        if len(assigned_users) < len(yelp.businesses) // 5:
            # Not enough users found via reviews: add high-activity users
            sorted_users = sorted(
                yelp.users.values(),
                key=lambda u: u.get("review_count", 0),
                reverse=True,
            )
            for u in sorted_users[: len(yelp.businesses) // 2]:
                assigned_users.add(u["user_id"])

        providers: dict[str, Provider] = {}
        for uid in assigned_users:
            u = yelp.users.get(uid)
            if u is None:
                continue
            avg_stars = float(u.get("average_stars", 3.0))
            elite = u.get("elite") or []
            if isinstance(elite, str):
                elite = [int(y) for y in elite.split(",") if y.strip().isdigit()]
            # Reputation: blend of average star rating and elite status
            elite_bonus = min(0.2, len(elite) * 0.02)
            reputation = round(min(1.0, _norm_stars(avg_stars) + elite_bonus), 3)
            providers[uid] = Provider(
                id=f"P_{uid[:10]}",
                name=u.get("name", "Unknown"),
                region=u.get("city", u.get("state", "UNKNOWN")),
                reputation=reputation,
                yelp_user_id=uid,
                elite_years=elite,
            )
        return providers, assigned_users

    # ------------------------------------------------------------------
    def _build_services_from_yelp(
        self,
        yelp: YelpLoader,
        providers: dict[str, Provider],
        resources: list[Resource],
    ) -> list[Service]:
        """Build one Service per Yelp business, assign IoT resources."""
        resource_ids = [r.id for r in resources]
        if not resource_ids:
            log.warning("No CASAS resources available — services will have no resources.")

        services = []
        # Build a fallback provider for businesses with no matched user
        fallback_providers = list(providers.values())

        for biz in yelp.businesses:
            bid = biz["business_id"]
            uid = yelp.business_top_user.get(bid)
            if uid and uid in providers:
                provider_id = providers[uid].id
            elif fallback_providers:
                # Deterministic fallback: hash business_id to pick a provider
                idx = _hash_seed(bid, self.seed) % len(fallback_providers)
                provider_id = fallback_providers[idx].id
            else:
                continue

            # Yelp QoS proxies
            stars = float(biz.get("stars", 3.0))
            review_count = int(biz.get("review_count", 0))
            is_open = bool(biz.get("is_open", 1))
            reliability = round(_norm_stars(stars) * 0.4 + 0.6, 4)  # never < 0.6
            availability = round(min(0.999, 0.80 + min(review_count / 5000.0, 0.19)), 4)

            # Resource assignment: pseudo-random, seeded from business_id
            if resource_ids:
                rng_local = random.Random(_hash_seed(bid, self.seed))
                k = rng_local.randint(self.min_res, self.max_res)
                assigned = rng_local.sample(resource_ids, k=min(k, len(resource_ids)))
            else:
                assigned = []

            category = _best_category(biz.get("categories"))
            services.append(
                Service(
                    id=f"S_{bid[:10]}",
                    name=biz.get("name", "Unknown"),
                    provider_id=provider_id,
                    category=category,
                    qos={
                        "response_time_ms": round(self.rng.uniform(10.0, 1500.0), 2),
                        "reliability": reliability,
                        "cost": round(self.rng.uniform(0.01, 40.0), 2),
                        "availability": availability,
                    },
                    resource_ids=assigned,
                    yelp_business_id=bid,
                    is_open=is_open,
                )
            )
        return services

    # ------------------------------------------------------------------
    def build(self) -> dict[str, Any]:
        # ----- 1. Load CASAS -----
        casas = CASASLoader(self.casas_dir).load() if self.casas_dir else CASASLoader(Path("/nonexistent"))

        # ----- 2. Load Yelp -----
        yelp = YelpLoader(self.yelp_dir) if self.yelp_dir else YelpLoader(Path("/nonexistent"))
        if self.yelp_dir and self.yelp_dir.exists():
            (yelp
             .load_businesses(max_businesses=self.max_businesses)
             .load_users(max_users=self.max_users)
             .build_business_provider_map(max_reviews=500_000))

        # ----- 3. Build entity lists -----
        log.info("Building IoT resources from CASAS sensors...")
        resources = self._build_resources_from_casas(casas)
        if not resources:
            log.warning("No CASAS resources built — TKG will have no IoT resources.")

        log.info("Building providers from Yelp users...")
        providers_by_uid, _ = self._build_providers_from_yelp(yelp)
        providers = list(providers_by_uid.values())

        log.info("Building IoT services from Yelp businesses...")
        resources_by_id = {r.id: r for r in resources}
        services = self._build_services_from_yelp(yelp, providers_by_uid, resources)

        log.info(
            f"  → {len(providers)} providers, {len(services)} services, {len(resources)} resources"
        )

        # ----- 4. Generate trust / conflict edges -----
        log.info("Generating trust/conflict/support edges...")
        edge_gen = TrustEdgeGenerator(
            seed=self.seed,
            conflict_density=self.conflict_density,
            provider_trust_density=self.provider_trust_density,
        )
        edges: list[dict] = []
        edges += edge_gen.offers_edges(services)
        edges += edge_gen.provider_trust_edges(providers)
        edges += edge_gen.service_resource_edges(services, resources_by_id)
        edges += edge_gen.allied_edges(services)
        edges += edge_gen.resource_conflict_edges(resources)
        log.info(f"  → {len(edges)} total edges generated.")

        # ----- 5. Assemble final dataset dict -----
        dataset = {
            "metadata": {
                "name": "hybrid_yelp_casas_tkg",
                "yelp_source": str(self.yelp_dir),
                "casas_source": str(self.casas_dir),
                "num_providers": len(providers),
                "num_services": len(services),
                "num_resources": len(resources),
                "num_edges": len(edges),
                "conflict_density": self.conflict_density,
                "provider_trust_density": self.provider_trust_density,
                "seed": self.seed,
            },
            "providers": [asdict(p) for p in providers],
            "services": [asdict(s) for s in services],
            "resources": [asdict(r) for r in resources],
            "edges": edges,
        }
        return dataset


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a hybrid Yelp + CASAS Trust Knowledge Graph (TKG) "
            "for Trust-MPGNN evaluation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("USAGE")[1].split("OUTPUT")[0],
    )
    p.add_argument(
        "--yelp-dir", default=None,
        help=(
            "Path to the extracted Yelp Open Dataset directory containing "
            "yelp_academic_dataset_business.json, "
            "yelp_academic_dataset_user.json, "
            "yelp_academic_dataset_review.json"
        ),
    )
    p.add_argument(
        "--casas-dir", default=None,
        help=(
            "Path to the CASAS dataset directory.  Can contain: "
            "(a) Kaggle CSV files (aruba.csv, milan.csv …), "
            "(b) legacy raw text files (aruba, hh101 …), or "
            "(c) modern sub-directories with event.csv + dataset.json."
        ),
    )
    p.add_argument("--output", default="hybrid_tkg.json",
                   help="Output JSON file path (default: hybrid_tkg.json)")
    p.add_argument("--max-businesses", type=int, default=2000,
                   help="Maximum number of Yelp businesses to load (default: 2000)")
    p.add_argument("--max-users", type=int, default=5000,
                   help="Maximum number of Yelp users to load (default: 5000)")
    p.add_argument("--resources-per-service", type=int, nargs=2, default=[1, 5],
                   metavar=("MIN", "MAX"),
                   help="Range of CASAS resources assigned per service (default: 1 5)")
    p.add_argument("--conflict-density", type=float, default=0.20,
                   help="Fraction of service-resource edges forced to OPPOSE (default: 0.20)")
    p.add_argument("--trust-density", type=float, default=0.08,
                   help="Fraction of provider-provider pairs given a TRUST edge (default: 0.08)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--no-indent", action="store_true",
                   help="Write compact JSON (no indentation) — faster for large outputs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Validate inputs
    if args.yelp_dir is None:
        log.warning(
            "No --yelp-dir provided. The builder will run but produce no "
            "services or providers.  Pass --yelp-dir /path/to/yelp_dataset "
            "to use the real data."
        )
    if args.casas_dir is None:
        log.warning(
            "No --casas-dir provided. The builder will run but produce no "
            "IoT resources.  Pass --casas-dir /path/to/casas_dataset "
            "to use the real data."
        )

    builder = HybridTKGBuilder(
        yelp_dir=args.yelp_dir,
        casas_dir=args.casas_dir,
        max_businesses=args.max_businesses,
        max_users=args.max_users,
        min_resources_per_service=args.resources_per_service[0],
        max_resources_per_service=args.resources_per_service[1],
        conflict_density=args.conflict_density,
        provider_trust_density=args.trust_density,
        seed=args.seed,
    )

    log.info("Starting Hybrid TKG construction...")
    dataset = builder.build()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.no_indent else 2
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=indent, ensure_ascii=False)

    meta = dataset["metadata"]
    log.info("=" * 55)
    log.info(f"Hybrid TKG saved → {out_path}")
    log.info(f"  Providers : {meta['num_providers']}")
    log.info(f"  Services  : {meta['num_services']}")
    log.info(f"  Resources : {meta['num_resources']}")
    log.info(f"  Edges     : {meta['num_edges']}")
    log.info(
        f"  File size : "
        f"{out_path.stat().st_size / 1_048_576:.1f} MB"
    )
    log.info("=" * 55)
    log.info(
        "The output file is directly consumable by Trust-MPGNN's "
        "tkg/builder.py (build_tkg_from_dataset(dataset_path=...))."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

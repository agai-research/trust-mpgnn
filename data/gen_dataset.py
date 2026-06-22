"""
gen_dataset.py - Generate the hybrid ICPS dataset (IoT services, resources, providers, trust/conflict relations)
Author: H. Mezni
"""

import json
import random
import uuid
import os
import sys
import logging
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# IoT domain knowledge
PROTOCOLS = ["Zigbee", "Bluetooth 5.0", "Wi-Fi 6", "Z-Wave", "MQTT", "CoAP", "LoRaWAN", "NB-IoT"]
DATA_FORMATS = ["JSON", "XML", "FHIR", "HL7", "CSV", "Protobuf", "CBOR"]
PRIVACY_POLICIES = ["GDPR", "HIPAA", "CCPA", "ISO27001", "NIST"]
CATEGORIES = ["smart_home", "smart_health", "smart_mobility", "smart_energy",
              "smart_security", "smart_education", "smart_retail", "smart_agriculture"]
SERVICE_TYPES = [
    "HealthTracker", "TemperatureMonitor", "OccupancyDetector", "EnergyManager",
    "SecurityCamera", "SmartLock", "LightController", "ClimateControl",
    "ParkingManager", "TrafficMonitor", "WasteManager", "WaterMonitor",
    "AirQualityMonitor", "NoiseMonitor", "ProductionTracker", "InventoryManager",
    "PatientMonitor", "MedicationTracker", "VitalSignsMonitor", "FitnessTracker",
    "SmartMeter", "EVCharger", "SolarMonitor", "HVACController",
    "AccessControl", "IntruderDetector", "FireDetector", "FloodSensor",
    "LearningAssistant", "AttendanceTracker", "SmartWhiteBoard", "ProjectorControl",
    "CheckoutMonitor", "StockTracker", "CustomerFlow", "PointOfSale",
    "IrrigationControl", "SoilMonitor", "GreenHouseControl", "LivestockTracker"
]
RESOURCE_TYPES = [
    "TemperatureSensor", "HumiditySensor", "MotionSensor", "OccupancySensor",
    "LightSensor", "SoundSensor", "PressureSensor", "ProximitySensor",
    "CameraDevice", "MicrophoneDevice", "SpeakerDevice", "DisplayDevice",
    "ActuatorDevice", "RelayDevice", "BuzzerDevice", "LEDStrip",
    "WearableDevice", "BioSensor", "Accelerometer", "GPS_Module",
    "SmartMeter_Res", "BatteryPack", "SolarPanel", "PowerGrid",
    "EdgeNode", "FogNode", "CloudGateway", "WiFiRouter",
    "SmartLock_Res", "DoorSensor", "WindowSensor", "SirenDevice",
    "SmartBoard", "Projector", "AudioHub", "MainScreen",
    "BarCodeScanner", "RFID_Reader", "PrinterDevice", "NFCReader",
    "IrrigationValve", "SoilProbe", "WeatherStation", "DroneController"
]
PROVIDER_NAMES = [
    "TechCorp", "SmartSys", "IoTHub", "ConnectPro", "SensorNet",
    "CloudIoT", "EdgeTech", "FogSystems", "SmartCity", "GreenTech",
    "HealthNet", "MobileSmart", "SecureIoT", "EduTech", "RetailSmart",
    "AgroSmart", "EnergySmart", "SafeNet", "DataStream", "OmniSense"
]


def generate_providers(n: int) -> list:
    """Generate IoT service providers with trust metadata."""
    providers = []
    for i in range(n):
        p = {
            "id": f"P{i:04d}",
            "name": f"{random.choice(PROVIDER_NAMES)}_{i}",
            "type": "Provider",
            "category": random.choice(CATEGORIES),
            "rating": round(random.uniform(2.5, 5.0), 2),
            "num_reviews": random.randint(10, 5000),
            "location": {"city": f"City_{i % 20}", "country": "TN"},
            "protocols": random.sample(PROTOCOLS, random.randint(1, 3)),
            "privacy_compliance": random.sample(PRIVACY_POLICIES, random.randint(1, 2)),
            "years_active": random.randint(1, 15)
        }
        providers.append(p)
    return providers


def generate_services(n: int, providers: list) -> list:
    """Generate IoT services with functional and QoS attributes."""
    services = []
    for i in range(n):
        provider = random.choice(providers)
        stype = random.choice(SERVICE_TYPES)
        s = {
            "id": f"S{i:05d}",
            "name": f"{stype}_{i}",
            "type": "Service",
            "category": provider["category"],
            "provider_id": provider["id"],
            "capabilities": random.sample(SERVICE_TYPES, random.randint(1, 4)),
            "protocols": random.sample(PROTOCOLS, random.randint(1, 3)),
            "data_format": random.choice(DATA_FORMATS),
            "privacy_compliance": random.sample(PRIVACY_POLICIES, random.randint(1, 2)),
            "qos": {
                "reliability": round(random.uniform(0.7, 1.0), 3),
                "response_time": round(random.uniform(10, 500), 1),
                "cost": round(random.uniform(0.01, 10.0), 3),
                "availability": round(random.uniform(0.9, 1.0), 3),
                "throughput": round(random.uniform(10, 1000), 1)
            },
            "feature_vector": [round(random.uniform(0, 1), 4) for _ in range(16)]
        }
        services.append(s)
    return services


def generate_resources(n: int, providers: list) -> list:
    """Generate IoT resources with physical and QoS attributes."""
    resources = []
    for i in range(n):
        provider = random.choice(providers)
        rtype = random.choice(RESOURCE_TYPES)
        r = {
            "id": f"R{i:05d}",
            "name": f"{rtype}_{i}",
            "type": "Resource",
            "resource_type": rtype,
            "provider_id": provider["id"],
            "protocols": random.sample(PROTOCOLS, random.randint(1, 3)),
            "data_format": random.choice(DATA_FORMATS),
            "privacy_compliance": random.sample(PRIVACY_POLICIES, random.randint(0, 2)),
            "energy_profile": round(random.uniform(0.1, 10.0), 2),
            "sensing_range": round(random.uniform(1, 100), 1),
            "qos": {
                "reliability": round(random.uniform(0.7, 1.0), 3),
                "response_time": round(random.uniform(5, 200), 1),
                "cost": round(random.uniform(0.001, 5.0), 3),
                "energy_efficiency": round(random.uniform(0.5, 1.0), 3)
            },
            "feature_vector": [round(random.uniform(0, 1), 4) for _ in range(16)]
        }
        resources.append(r)
    return resources


def is_compatible(e1: dict, e2: dict) -> bool:
    """Check protocol/format compatibility between two entities."""
    shared_protocols = set(e1.get("protocols", [])) & set(e2.get("protocols", []))
    shared_privacy = set(e1.get("privacy_compliance", [])) & set(e2.get("privacy_compliance", []))
    return len(shared_protocols) > 0 or len(shared_privacy) > 0


def generate_relations(providers: list, services: list, resources: list,
                        target_total: int, conflict_density: float) -> list:
    """
    Generate trust and conflict relations among ICPS entities.
    Relation types: TRUST (P-P), SUPPORT/OPPOSE/NEUTRAL (S-R), ALLIED (S-S), CONFLICT (R-R).
    conflict_density controls percentage of OPPOSE/CONFLICT edges.
    """
    relations = []
    pid = {p["id"]: p for p in providers}
    sid = {s["id"]: s for s in services}
    rid = {r["id"]: r for r in resources}

    n_trust = int(target_total * 0.25)
    n_sr = int(target_total * 0.40)
    n_allied = int(target_total * 0.20)
    n_rr = int(target_total * 0.15)

    # Provider-Provider TRUST relations
    pairs_used = set()
    while len([r for r in relations if r["type"] == "TRUST"]) < n_trust:
        pi = random.choice(providers)
        pj = random.choice(providers)
        if pi["id"] == pj["id"] or (pi["id"], pj["id"]) in pairs_used:
            continue
        pairs_used.add((pi["id"], pj["id"]))
        relations.append({"head": pi["id"], "relation": "TRUST", "tail": pj["id"],
                           "type": "TRUST", "head_type": "Provider", "tail_type": "Provider"})

    # Service-Resource SUPPORT/OPPOSE/NEUTRAL
    pairs_used = set()
    while len([r for r in relations if r["head_type"] == "Service"]) < n_sr:
        s = random.choice(services)
        r = random.choice(resources)
        if (s["id"], r["id"]) in pairs_used:
            continue
        pairs_used.add((s["id"], r["id"]))
        compat = is_compatible(s, r)
        rand = random.random()
        if rand < conflict_density:
            rel = "OPPOSE"
        elif compat:
            rel = "SUPPORT"
        else:
            rel = "NEUTRAL"
        relations.append({"head": s["id"], "relation": rel, "tail": r["id"],
                           "type": rel, "head_type": "Service", "tail_type": "Resource"})

    # Service-Service ALLIED/CONFLICT relations
    pairs_used = set()
    while len([r for r in relations if r["head_type"] == "Service" and r["tail_type"] == "Service"]) < n_allied:
        si = random.choice(services)
        sj = random.choice(services)
        if si["id"] == sj["id"] or (si["id"], sj["id"]) in pairs_used:
            continue
        pairs_used.add((si["id"], sj["id"]))
        compat = is_compatible(si, sj)
        rand = random.random()
        if rand < conflict_density:
            rel = "CONFLICT"
        else:
            rel = "ALLIED"
        relations.append({"head": si["id"], "relation": rel, "tail": sj["id"],
                           "type": rel, "head_type": "Service", "tail_type": "Service"})

    # Resource-Resource CONFLICT/TRUST
    pairs_used = set()
    while len([r for r in relations if r["head_type"] == "Resource"]) < n_rr:
        ri = random.choice(resources)
        rj = random.choice(resources)
        if ri["id"] == rj["id"] or (ri["id"], rj["id"]) in pairs_used:
            continue
        pairs_used.add((ri["id"], rj["id"]))
        compat = is_compatible(ri, rj)
        rand = random.random()
        if rand < conflict_density:
            rel = "CONFLICT"
        else:
            rel = "TRUST"
        relations.append({"head": ri["id"], "relation": rel, "tail": rj["id"],
                           "type": rel, "head_type": "Resource", "tail_type": "Resource"})

    return relations


def generate_dataset(num_providers: int, num_services: int, num_resources: int,
                      num_relations: int, conflict_density: float = 0.2,
                      seed: int = 42) -> dict:
    """Build the full ICPS hybrid dataset."""
    random.seed(seed)
    logger.info("Generating providers ...")
    providers = generate_providers(num_providers)
    logger.info("Generating services ...")
    services = generate_services(num_services, providers)
    logger.info("Generating resources ...")
    resources = generate_resources(num_resources, providers)
    logger.info("Generating trust/conflict relations ...")
    relations = generate_relations(providers, services, resources, num_relations, conflict_density)

    dataset = {
        "metadata": {
            "num_providers": len(providers),
            "num_services": len(services),
            "num_resources": len(resources),
            "num_relations": len(relations),
            "conflict_density": conflict_density,
            "seed": seed,
            "author": "H. Mezni"
        },
        "providers": providers,
        "services": services,
        "resources": resources,
        "relations": relations
    }
    # Relation type summary
    from collections import Counter
    rcnt = Counter(r["type"] for r in relations)
    dataset["metadata"]["relation_counts"] = dict(rcnt)
    logger.info(f"Dataset generated: {len(providers)} providers, {len(services)} services, "
                f"{len(resources)} resources, {len(relations)} relations.")
    logger.info(f"Relation distribution: {dict(rcnt)}")
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Generate the hybrid ICPS dataset")
    parser.add_argument("--providers", type=int, default=500)
    parser.add_argument("--services", type=int, default=2000)
    parser.add_argument("--resources", type=int, default=1000)
    parser.add_argument("--relations", type=int, default=5000)
    parser.add_argument("--conflict_density", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/raw/dataset.json")
    args = parser.parse_args()

    dataset = generate_dataset(
        args.providers, args.services, args.resources,
        args.relations, args.conflict_density, args.seed
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dataset, f, indent=2)
    logger.info(f"Dataset saved to {args.out}")


if __name__ == "__main__":
    main()

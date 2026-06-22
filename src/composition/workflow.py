"""
workflow.py - IoT workflow representation and query parsing.
Defines the abstract IoT workflow W_u and user query format.
Author: H. Mezni
"""

import json
import random
import numpy as np
from typing import List, Dict


def random_feature_vector(dim: int = 16) -> List[float]:
    """Generate a random normalized feature vector."""
    v = np.random.uniform(0, 1, dim)
    v = v / (np.linalg.norm(v) + 1e-8)
    return v.tolist()


def create_workflow(tasks: List[Dict]) -> List[Dict]:
    """
    Create an abstract IoT workflow from a list of task descriptors.
    Each task: {task_id, task_name, task_features (optional), resource_features (optional)}
    """
    workflow = []
    for i, task in enumerate(tasks):
        workflow.append({
            "task_id": task.get("task_id", f"T{i+1}"),
            "task_name": task.get("task_name", f"Task_{i+1}"),
            "task_features": task.get("task_features", random_feature_vector(16)),
            "resource_features": task.get("resource_features", random_feature_vector(16))
        })
    return workflow


def load_workflow_from_file(path: str) -> List[Dict]:
    """Load a workflow from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return data.get("workflow", data)


def generate_workflow(size: int, seed: int = 42) -> List[Dict]:
    """Generate a synthetic IoT workflow of given size."""
    random.seed(seed)
    np.random.seed(seed)
    task_templates = [
        "HealthMonitor", "TemperatureControl", "AccessControl", "EnergyManage",
        "SecurityMonitor", "LightControl", "OccupancyDetect", "AirQuality",
        "TrafficMonitor", "ParkingManage", "WasteManage", "WaterMonitor",
        "ProductionTrack", "InventoryManage", "PatientMonitor", "FitnessTrack"
    ]
    tasks = []
    for i in range(size):
        name = f"{random.choice(task_templates)}_{i+1}"
        tasks.append({
            "task_id": f"T{i+1:02d}",
            "task_name": name,
            "task_features": random_feature_vector(16),
            "resource_features": random_feature_vector(16)
        })
    return create_workflow(tasks)


# --- Predefined test queries ---
SAMPLE_QUERIES = [
    {
        "query_id": "Q1",
        "description": "Smart healthcare monitoring application",
        "workflow": [
            {"task_id": "T1", "task_name": "VitalSigns", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T2", "task_name": "MedicationTracker", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T3", "task_name": "FitnessTracker", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)}
        ]
    },
    {
        "query_id": "Q2",
        "description": "Smart building energy management",
        "workflow": [
            {"task_id": "T1", "task_name": "EnergyMonitor", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T2", "task_name": "ClimateControl", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T3", "task_name": "LightControl", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T4", "task_name": "OccupancyDetect", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)}
        ]
    },
    {
        "query_id": "Q3",
        "description": "Smart mobility and parking",
        "workflow": [
            {"task_id": "T1", "task_name": "ParkingMonitor", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T2", "task_name": "TrafficControl", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T3", "task_name": "EVCharging", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)}
        ]
    },
    {
        "query_id": "Q4",
        "description": "Smart home automation",
        "workflow": [
            {"task_id": "T1", "task_name": "SecurityCam", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T2", "task_name": "SmartLock", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T3", "task_name": "AmbientLight", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T4", "task_name": "ThermostatCtrl", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)},
            {"task_id": "T5", "task_name": "FloodDetect", "task_features": random_feature_vector(16), "resource_features": random_feature_vector(16)}
        ]
    }
]

"""
metapaths.py - Trust meta-path definitions for the ICPS TKG.
Based on Section 4.2 of the Trust-MPGNN paper.
Author: H. Mezni
"""

# Each metapath is defined as a list of relation types to follow in order.
# Name -> (path of relations, description)

METAPATHS = {
    "PP_trust": {
        "relations": ["TRUST"],
        "node_types": ["Provider", "Provider"],
        "description": "Provider-TRUST-Provider: direct provider trust"
    },
    "PP_trust2": {
        "relations": ["TRUST", "TRUST"],
        "node_types": ["Provider", "Provider", "Provider"],
        "description": "Provider-TRUST-Provider-TRUST-Provider: transitive provider trust"
    },
    "SR_support": {
        "relations": ["SUPPORT"],
        "node_types": ["Service", "Resource"],
        "description": "Service-SUPPORT-Resource: service supports resource usage"
    },
    "SR_oppose": {
        "relations": ["OPPOSE"],
        "node_types": ["Service", "Resource"],
        "description": "Service-OPPOSE-Resource: service opposes resource (conflict)"
    },
    "SS_allied": {
        "relations": ["ALLIED"],
        "node_types": ["Service", "Service"],
        "description": "Service-ALLIED-Service: allied services coalition"
    },
    "SRS": {
        "relations": ["SUPPORT", "SUPPORT"],
        "node_types": ["Service", "Resource", "Service"],
        "description": "Service-SUPPORT-Resource-SUPPORT-Service: mutual resource support"
    },
    "PSR": {
        "relations": ["TRUST", "SUPPORT"],
        "node_types": ["Provider", "Provider", "Resource"],
        "description": "Provider-TRUST-Provider then SERVICE-SUPPORT-Resource trust chain"
    },
    "PPS": {
        "relations": ["TRUST", "TRUST"],
        "node_types": ["Provider", "Provider", "Provider"],
        "description": "Provider-TRUST-Provider-TRUST: extended provider trust"
    },
    "PSS": {
        "relations": ["ALLIED", "ALLIED"],
        "node_types": ["Service", "Service", "Service"],
        "description": "Service-ALLIED-Service-ALLIED-Service: service coalition chain"
    },
    "RR_conflict": {
        "relations": ["CONFLICT"],
        "node_types": ["Resource", "Resource"],
        "description": "Resource-CONFLICT-Resource: resource conflict detection"
    }
}


def get_metapath_names() -> list:
    return list(METAPATHS.keys())


def get_metapath_relations(name: str) -> list:
    return METAPATHS[name]["relations"]


def get_all_relations() -> list:
    """Return flat list of all unique relation sequences used in metapaths."""
    return [v["relations"] for v in METAPATHS.values()]

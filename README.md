# Trust-MPGNN: Conflict-aware Composition of IoT Services

> **Metapath-Guided Trust Learning for Intelligent Cyber-Physical Systems**
> Author: **H. Mezni**

---

## Overview

**Trust-MPGNN** is a conflict- and trust-aware IoT service composition framework built around three key phases:

1. **Trust Knowledge Graph (TKG) Construction** — Formalizes trust and conflict relations among ICPS entities (providers, IoT services, IoT resources) as a directed heterogeneous graph.
2. **Metapath-Guided Trust Learning (Algorithm 1)** — Uses a GNN variant with attention-based aggregation along predefined trust meta-paths to learn node embeddings and predict latent trust/conflict relations.
3. **Trust-Aware IoT Service Composition (Algorithm 2)** — Selects and composes non-conflicting, trustworthy IoT services and resources matching user workflow requirements via cosine-similarity matching in the trust embedding space.

---

## Project Structure

```
Trust-MPGNN/
├── config.json                  # All hyperparameters and paths
├── trust-mpgnn.py               # Main prototype runner (full pipeline)
├── main-tkg.py                  # TKG construction entry point
├── main-exp.py                  # Generate all experiment instances
│
├── src/
│   ├── tkg/
│   │   ├── tkg.py               # TrustKnowledgeGraph class (NetworkX)
│   │   └── metapaths.py         # Trust meta-path definitions
│   ├── gnn/
│   │   ├── model.py             # TrustMPGNN model (GNN + attention + MLP predictor)
│   │   ├── sampler.py           # Metapath-guided neighborhood sampler
│   │   └── trainer.py           # Training loop (Algorithm 1)
│   ├── composition/
│   │   ├── composer.py          # IoT service composer (Algorithm 2)
│   │   ├── workflow.py          # Workflow/query representation
│   │   └── baselines.py         # Baseline methods (Trust-GNN, FFCA-IoTSC, TQoSC)
│   └── utils/
│       └── utils.py             # Logging, JSON I/O, path utilities
│
├── data/
│   ├── gen_dataset.py           # ICPS hybrid dataset generator
│   ├── split_dataset.py         # Creates experiment-specific instances
│   ├── raw/
│   │   └── dataset.json         # Main generated dataset (auto-created)
│   └── instances/               # Per-experiment TKG instances (auto-created)
│       ├── tkg_size_1000.json
│       ├── tkg_density_10.json
│       └── ...
│
├── experiments/
│   ├── exp_conflict_density.py  # Exp 1: Quality vs conflict density (Table 8)
│   ├── exp_workflow.py          # Exp 2: Quality vs workflow complexity (Table 9)
│   ├── exp_threshold.py         # Exp 3: Trust threshold sensitivity (Table 10)
│   ├── exp_icps_size.py         # Exp 4: Composition time vs ICPS size (Figure)
│   └── exp_time_density.py      # Exp 5: Composition time vs conflict density (Table 11)
│
├── tests/
│   └── test_prototype.py        # First-test: validate prototype on small TKG
│
├── web/
│   └── index.html               # Interactive web UI (TKG view + composition)
│
├── output/                      # Generated results (auto-created)
│   ├── tkg.json
│   ├── embeddings.pt
│   ├── composition_result.json
│   └── exp_*.json
│
└── logs/                        # Execution logs (auto-created)
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NetworkX 3.1+

### Setup

```bash
# Clone or copy the project
cd Trust-MPGNN

# Install dependencies
pip install torch networkx numpy

# Optional (for RDF export)
pip install rdflib
```

---

## Quick Start

### Step 1 — Generate the dataset and experiment instances

```bash
python main-exp.py
```

This runs `data/gen_dataset.py` (builds the hybrid ICPS dataset with 500 providers, 2000 services, 1000 resources) then `data/split_dataset.py` (creates per-experiment TKG instances).

### Step 2 — Build the TKG

```bash
python main-tkg.py
```

Loads the dataset, constructs the TKG using NetworkX, and exports `output/tkg.json`.

### Step 3 — Run the prototype

```bash
# Default run (workflow size=10)
python trust-mpgnn.py

# Custom workflow size
python trust-mpgnn.py --workflow 20

# Use a predefined sample query (Q1–Q4)
python trust-mpgnn.py --query 2

# Set a custom trust threshold
python trust-mpgnn.py --theta 0.7

# Use a specific dataset instance
python trust-mpgnn.py --dataset data/instances/tkg_density_30.json
```

### Step 4 — Run the first validation test

```bash
python tests/test_prototype.py
```

Tests all 4 predefined IoT queries on a small TKG (200 nodes). Results saved to `output/test_results.json`.

---

## Running Experiments

All experiment scripts are in `experiments/`. Each script loads the appropriate dataset instance, trains Trust-MPGNN, runs baselines, and saves results as JSON.

```bash
# Exp 1: Impact of conflict density (Table 8)
python experiments/exp_conflict_density.py

# Exp 2: Workflow complexity impact (Table 9)
python experiments/exp_workflow.py

# Exp 3: Trust threshold sensitivity (Table 10)
python experiments/exp_threshold.py

# Exp 4: Scalability — ICPS size (Figure in paper)
python experiments/exp_icps_size.py

# Exp 5: Composition time vs conflict density (Table 11)
python experiments/exp_time_density.py
```

Results are saved in `output/exp_*.json`.

---

## Configuration (`config.json`)

All hyperparameters are defined in `config.json`:

| Parameter | Default | Description |
|---|---|---|
| `gnn.embed_dim` | 128 | Node embedding dimension |
| `gnn.num_layers` | 2 | Number of GNN layers |
| `gnn.dropout` | 0.3 | Dropout rate |
| `gnn.learning_rate` | 0.001 | Adam optimizer learning rate |
| `gnn.epochs` | 200 | Training epochs |
| `gnn.trust_threshold` | 0.6 | Theta: trust vs conflict boundary |
| `gnn.sampling_size` | 20 | Max neighbors per metapath (K) |
| `gnn.batch_size` | 256 | Mini-batch size |
| `tkg.num_providers` | 500 | Providers in main dataset |
| `tkg.num_services` | 2000 | IoT services |
| `tkg.num_resources` | 1000 | IoT resources |
| `tkg.num_relations` | 5000 | Trust/conflict edges |

---

## Trust Meta-Paths

Ten meta-paths are predefined in `src/tkg/metapaths.py`, covering the key trust propagation patterns:

| Name | Path | Description |
|---|---|---|
| `PP_trust` | Provider→TRUST→Provider | Direct provider trust |
| `PP_trust2` | P→TRUST→P→TRUST→P | Transitive provider trust |
| `SR_support` | Service→SUPPORT→Resource | Service supports resource |
| `SR_oppose` | Service→OPPOSE→Resource | Service opposes resource |
| `SS_allied` | Service→ALLIED→Service | Allied service coalition |
| `SRS` | S→SUPPORT→R→SUPPORT→S | Mutual resource support |
| `PSR` | P→TRUST→P→SUPPORT→R | Provider-mediated resource trust |
| `PPS` | P→TRUST→P→TRUST→P | Extended provider trust |
| `PSS` | S→ALLIED→S→ALLIED→S | Service coalition chain |
| `RR_conflict` | Resource→CONFLICT→Resource | Resource conflict detection |

---

## TKG Relation Types

| Relation | Head → Tail | Meaning |
|---|---|---|
| `TRUST` | Provider → Provider | Provider-level bilateral trust |
| `SUPPORT` | Service → Resource | Service endorses resource usage |
| `OPPOSE` | Service → Resource | Service opposes resource sharing |
| `NEUTRAL` | Service → Resource | No explicit trust/conflict |
| `ALLIED` | Service → Service | Services form a trusted coalition |
| `CONFLICT` | Resource → Resource | Resources are incompatible |

---

## Web Interface

Open `web/index.html` in any modern browser (no server required):

- **Left panel**: Select from 4 predefined IoT queries
- **Right top**: Interactive TKG visualization (drag nodes, zoom)
- **Right bottom**: Click **Compose** to see assigned services, resources, and trust metrics

---

## Output Files

| File | Description |
|---|---|
| `output/tkg.json` | TKG graph structure (nodes + edges) |
| `output/embeddings.pt` | Learned trust embedding space (PyTorch) |
| `output/composition_result.json` | Composition output for the main run |
| `output/test_results.json` | First-test query results |
| `output/exp_conflict_density.json` | Experiment 1 results |
| `output/exp_workflow.json` | Experiment 2 results |
| `output/exp_threshold.json` | Experiment 3 results |
| `output/exp_icps_size.json` | Experiment 4 results |
| `output/exp_time_density.json` | Experiment 5 results |

---

## Reference

> F. Ghedass, H. Mezni, M. Alabdulhafith, H. Elmannai.
> *Conflict-aware Composition of IoT Services: an Approach based on MetaPath-Guided Trust Learning.*
> Internet of Things, Elsevier, 2026.

---

## License

For academic and research use. Author: **H. Mezni**, Jendouba University, Tunisia.

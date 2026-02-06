# SNNFrame Format Reference

Complete specification for all four supported network description formats.

## Format Overview

| Format | Extension | Type | Best For |
|--------|-----------|------|----------|
| **Native JSON** | `.snnf.json` | JSON | New projects; full SNNFrame feature access |
| **SONATA** | `circuit_config.json`, `.sonata.json` | JSON + HDF5 | Large-scale models from Blue Brain / Allen Institute |
| **NeuroML** | `.nml`, `.neuroml` | XML | Interoperability with the NeuroML ecosystem |
| **HOC** | `.hoc` | Script | Importing existing NEURON simulator models |

All formats are parsed into a common `NetworkIR` intermediate representation, then constructed into a running SNNFrame network.

---

## Native JSON Format (`.snnf.json`)

The most expressive format, designed specifically for SNNFrame.

### Top-Level Structure

```json
{
  "snnframe_version": "1.0",
  "neuron_params": { ... },
  "input_layer": { ... },
  "output_layer": { ... },
  "brain": { ... },
  "projections": [ ... ],
  "gabor": { ... },
  "saccades": { ... },
  "simulation": { ... }
}
```

### Neuron Parameter Sets

Define reusable parameter profiles:

```json
"neuron_params": {
  "cortical_default": {
    "window_size_ms": 500.0,
    "similarity_threshold": 0.93,
    "max_reference_patterns": 500,
    "similarity_metric": "cosine"
  },
  "output_neuron": {
    "window_size_ms": 500.0,
    "similarity_threshold": 0.85,
    "max_reference_patterns": 200
  }
}
```

Parameters are referenced by name in population definitions.

### Input Layer

```json
"input_layer": {
  "name": "InputGrid",
  "rows": 28,
  "cols": 28,
  "latency_ms": 15.0,
  "pixel_threshold": 0.4,
  "neuron_params": {
    "window_size_ms": 100.0,
    "similarity_threshold": 0.5,
    "max_reference_patterns": 50
  }
}
```

### Output Layer

```json
"output_layer": {
  "name": "OutputLayer",
  "num_classes": 26,
  "neurons_per_class": 3,
  "neuron_params": {
    "window_size_ms": 500.0,
    "similarity_threshold": 0.85,
    "max_reference_patterns": 200
  }
}
```

### Brain Hierarchy

The hierarchy follows: Brain → Hemispheres → Lobes → Regions → Nuclei → Columns → Layers → Populations.

```json
"brain": {
  "name": "MyBrain",
  "hemispheres": [{
    "name": "Left",
    "lobes": [{
      "name": "Occipital",
      "regions": [{
        "name": "V1",
        "nuclei": [{
          "name": "FeatureColumns",
          "columns": [
            { "name": "Col1", "layers": [...] }
          ]
        }]
      }]
    }]
  }]
}
```

### Column Templates

Instead of defining each column individually, use a template to generate multiple columns:

```json
"column_template": {
  "template_name": "CorticalColumn",
  "naming_pattern": "Orient_{orientation}_Freq_{frequency}",
  "orientations": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
  "frequencies": [3.0, 8.0],
  "layers": [
    {
      "name": "L4",
      "populations": [
        { "name": "L4_stellate", "count": 49, "neuron_params": "cortical_default", "grid_layout": "7x7" }
      ]
    },
    {
      "name": "L5",
      "populations": [
        { "name": "L5_pyramidal", "count": 80, "neuron_params": "cortical_default" }
      ]
    }
  ]
}
```

This generates `len(orientations) × len(frequencies)` columns (e.g., 8 × 2 = 16).

### Populations

```json
{
  "name": "L4_stellate",
  "count": 49,
  "neuron_params": "cortical_default",
  "grid_layout": "7x7"
}
```

- `name` — Population identifier
- `count` — Number of neurons
- `neuron_params` — Reference to a named parameter set
- `grid_layout` (optional) — Spatial arrangement (e.g., "7x7" for topographic mapping)

### Projections

```json
"projections": [
  {
    "name": "L4_to_L5",
    "source": "V1/*/L4",
    "target": "V1/*/L5",
    "pattern": "random_sparse",
    "probability": 0.25,
    "weight": 0.1,
    "max_weight": 0.5,
    "delay": 1.0,
    "scope": "intra_column",
    "synapse_group": "L4ToL5"
  }
]
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Projection identifier |
| `source` | string | Source path (population name or glob path like `V1/*/L4`) |
| `target` | string | Target path (population name or glob path) |
| `pattern` | string | Connectivity pattern (see below) |
| `probability` | float | Connection probability (for random_sparse) |
| `weight` | float | Initial synaptic weight |
| `max_weight` | float | Maximum synaptic weight |
| `delay` | float | Transmission delay in ms |
| `scope` | string | `"global"` (all-to-all columns) or `"intra_column"` (same column only) |
| `synapse_group` | string | Tag for differential STDP treatment |

**Connectivity Patterns:**
- `random_sparse` — Connect with given probability
- `all_to_all` — Every source to every target
- `one_to_one` — 1:1 mapping by index
- `many_to_one` — Convergent connections
- `distance_dependent` — Gaussian falloff with distance
- `topographic` — Spatial relationship preservation
- `small_world` — Local clusters + long-range connections

**Path Syntax:**
- `"InputGrid"` — References the input layer
- `"OutputLayer"` — References the output layer
- `"V1/*/L4"` — All L4 layers across all columns in V1 region
- `*` matches any column name (one level)

### Gabor Configuration

```json
"gabor": {
  "freq_low": 8.0,
  "freq_high": 3.0,
  "sigma": 2.0,
  "gamma": 0.5,
  "threshold": 0.5,
  "kernel_size": 7
}
```

### Saccade Configuration

```json
"saccades": {
  "enabled": true,
  "num_fixations": 4,
  "regions": [
    { "name": "top",    "row_start": 0,  "row_end": 13, "col_start": 0, "col_end": 27 },
    { "name": "bottom", "row_start": 14, "row_end": 27, "col_start": 0, "col_end": 27 },
    { "name": "center", "row_start": 7,  "row_end": 20, "col_start": 7, "col_end": 20 },
    { "name": "full",   "row_start": 0,  "row_end": 27, "col_start": 0, "col_end": 27 }
  ]
}
```

### Simulation Configuration

```json
"simulation": {
  "spike_processor": {
    "time_slices": 10000,
    "threads": 24,
    "real_time_sync": false
  },
  "stdp": {
    "enabled": true,
    "ltd_scale": 0.3,
    "ltd_window_ms": 70.0,
    "trace_stdp": true,
    "freeze_during_testing": true
  },
  "competition": {
    "l4_keep": 8,
    "l5_keep": 8,
    "enable_l5_inhibition": true
  },
  "inter_image_gap_ms": 550.0
}
```

### Complete Example

See `configs/emnist_v1_network.snnf.json` for a full example implementing the EMNIST letters classifier.

---

## SONATA Format (`circuit_config.json`)

The [SONATA](https://github.com/AllenInstitute/sonata) format from the Blue Brain Project / Allen Institute stores networks in HDF5 files with a JSON configuration file.

### File Structure

```
my_model/
├── circuit_config.json    ← Entry point (parsed by SONATAParser)
└── networks/
    ├── nodes.h5           ← Node populations (HDF5)
    ├── node_types.csv     ← Node type metadata
    ├── edges.h5           ← Edge populations (HDF5)
    └── edge_types.csv     ← Edge type metadata
```

### circuit_config.json

```json
{
  "network_name": "MySONATANetwork",
  "manifest": {
    "$BASE_DIR": ".",
    "$NETWORK_DIR": "$BASE_DIR/networks"
  },
  "networks": {
    "nodes": [
      { "nodes_file": "$NETWORK_DIR/nodes.h5", "node_types_file": "$NETWORK_DIR/node_types.csv" }
    ],
    "edges": [
      { "edges_file": "$NETWORK_DIR/edges.h5", "edge_types_file": "$NETWORK_DIR/edge_types.csv" }
    ]
  },
  "snnframe": {
    "neuron_params": {
      "cortical": { "window_size_ms": 200.0, "similarity_threshold": 0.93 }
    },
    "input_layer": { "rows": 14, "cols": 14, "latency_ms": 15.0 },
    "output_layer": { "num_classes": 26, "neurons_per_class": 3 },
    "gabor": { "freq_low": 8.0, "freq_high": 3.0, "sigma": 2.0, "gamma": 0.5, "threshold": 0.5, "kernel_size": 7 }
  }
}
```

### Manifest Variables

Variables like `$BASE_DIR` and `$NETWORK_DIR` are resolved before use:
- `$BASE_DIR` is resolved relative to the directory containing `circuit_config.json`
- Variables can reference other variables (e.g., `$NETWORK_DIR` references `$BASE_DIR`)

### HDF5 Node Files

Nodes are stored under `/nodes/<population>/` groups:

| Dataset | Type | Description |
|---------|------|-------------|
| `node_type_id` | int[] | Type identifier per node |
| `node_id` | int[] | Unique node identifier |
| Custom attributes | varies | SNNFrame properties (window_size_ms, etc.) |

### HDF5 Edge Files

Edges are stored under `/edges/<population>/` groups:

| Dataset | Type | Description |
|---------|------|-------------|
| `source_node_id` | int[] | Source node index |
| `target_node_id` | int[] | Target node index |
| `edge_type_id` | int[] | Edge type identifier |
| `weight` | float[] | Synaptic weight |
| `delay` | float[] | Transmission delay (ms) |

### SNNFrame Extensions

The `"snnframe"` section in `circuit_config.json` provides SNNFrame-specific configuration that is not part of the standard SONATA spec. It supports `neuron_params`, `input_layer`, `output_layer`, and `gabor` subsections.

### Complete Example

See `configs/example_sonata/circuit_config.json` for a working example.

---

## NeuroML Format (`.nml`, `.neuroml`)

[NeuroML](https://neuroml.org/) is an XML-based community standard for describing neural models. SNNFrame's `NeuroMLParser` reads NeuroML v2 files and maps their structure to the NetworkIR.

### Document Structure

```xml
<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2"
         id="my_network">

    <!-- Cell type definitions -->
    <cell id="..."> ... </cell>

    <!-- Synapse type definitions -->
    <expOneSynapse id="..." ...> ... </expOneSynapse>

    <!-- Network definition -->
    <network id="...">
        <population id="..." component="..." size="..."/>
        <projection id="..." presynapticPopulation="..." postsynapticPopulation="..." synapse="...">
            <connection id="..." preCellId="..." postCellId="..."/>
        </projection>
    </network>

</neuroml>
```

### Cell Types

Cell types define neuron parameters. SNNFrame-specific properties use the `snnfw:` namespace prefix:

```xml
<cell id="cortical_cell">
    <property tag="snnfw:window_size_ms" value="200.0"/>
    <property tag="snnfw:similarity_threshold" value="0.93"/>
    <property tag="snnfw:max_reference_patterns" value="500"/>
    <property tag="snnfw:similarity_metric" value="cosine"/>
</cell>
```

**Supported `snnfw:` properties:**

| Property | Type | Description |
|----------|------|-------------|
| `snnfw:window_size_ms` | float | Spike collection window (ms) |
| `snnfw:similarity_threshold` | float | Pattern match threshold (0.0–1.0) |
| `snnfw:max_reference_patterns` | int | Maximum learned patterns |
| `snnfw:similarity_metric` | string | `cosine`, `histogram`, `euclidean`, `correlation`, `waveform` |

### Synapse Types

Synapse types carry weight and delay information:

```xml
<expOneSynapse id="excitatory_syn" tauDecay="5ms" gbase="0.8nS" erev="0mV">
    <property tag="snnfw:weight" value="0.5"/>
    <property tag="snnfw:delay" value="1.5"/>
</expOneSynapse>
```

### Populations

Populations reference a cell type and specify the number of neurons:

```xml
<population id="L4_neurons" component="cortical_cell" size="49">
    <property tag="snnfw:layer" value="L4"/>
</population>
```

The `snnfw:layer` property is used by the parser to organize populations into SNNFrame's layer hierarchy.

### Projections and Connections

Projections connect two populations through a named synapse:

```xml
<projection id="L4_to_L5"
            presynapticPopulation="L4_neurons"
            postsynapticPopulation="L5_neurons"
            synapse="excitatory_syn">
    <connection id="0" preCellId="../L4_neurons[0]" postCellId="../L5_neurons[0]"/>
    <connection id="1" preCellId="../L4_neurons[1]" postCellId="../L5_neurons[1]"/>
</projection>
```

- The `synapse` attribute references a synapse type for weight/delay defaults
- Individual `<connection>` elements define explicit cell-to-cell connectivity
- Cell IDs use NeuroML path format: `../<population>[<index>]`
- If no explicit connections are listed, the parser generates `one_to_one` connectivity

### Mapping to SNNFrame

| NeuroML Element | SNNFrame Equivalent |
|-----------------|-------------------|
| `<cell>` | `NeuronParamsIR` entry |
| `<population>` | `PopulationIR` within a generated layer/column |
| `<projection>` | `ProjectionIR` with pattern and weight |
| `<connection>` | Individual synapse within a projection |
| `<network>` | Top-level `BrainIR` structure |

### Complete Example

See `configs/example_network.nml` for a working example.

---

## HOC Format (`.hoc`)

[HOC](https://nrn.readthedocs.io/en/latest/hoc/index.html) is the scripting language of the NEURON simulator. SNNFrame's `HOCParser` extracts declarative information from procedural HOC scripts.

### Supported Constructs

The HOC parser recognizes the following patterns:

#### Cell Templates

```hoc
begintemplate CorticalL4
    proc init() {
        window_size_ms = 200
        similarity_threshold = 0.93
        max_reference_patterns = 500
    }
    create soma, axon, dendrite
endtemplate CorticalL4
```

The parser extracts:
- Template name → neuron parameter set name and population type
- Variables in `init()` → neuron parameters (window_size_ms, similarity_threshold, max_reference_patterns)
- `create` statements → compartment metadata (informational)

#### Cell Instantiation

```hoc
objref cells
cells = new List()

for i = 0, 48 {
    cells.append(new CorticalL4())
}
```

The parser extracts:
- Loop count → population size (49 neurons: indices 0 through 48)
- Template reference → links population to neuron parameters

#### Connections (NetCon)

```hoc
for i = 0, 48 {
    nc = new NetCon(l4_cells.o(i).soma(0.5), l5_cells.o(i).syn, 0, 1.5, 0.5)
}
```

**NetCon arguments:** `NetCon(source, target, threshold, delay, weight)`

The parser extracts:
- Source/target list names → source and target populations
- `delay` (4th argument) → projection delay in ms
- `weight` (5th argument) → projection weight
- Loop count → connectivity pattern (inferred as one_to_one or all_to_all)

### Limitations

Because HOC is a procedural language, the parser applies heuristics:

| Feature | Support Level |
|---------|--------------|
| `begintemplate`/`endtemplate` | ✅ Full |
| `for` loop instantiation | ✅ Full |
| `new NetCon(...)` connections | ✅ Full |
| Conditional logic (`if`/`else`) | ⚠️ Ignored (all branches treated as executed) |
| Function calls / procedures | ⚠️ Only `init()` variables extracted |
| File includes (`load_file`) | ❌ Not followed |
| Dynamic variable computation | ❌ Not evaluated |

### Mapping to SNNFrame

| HOC Construct | SNNFrame Equivalent |
|---------------|-------------------|
| `begintemplate` | `NeuronParamsIR` entry |
| `for i = 0, N { list.append(new Template()) }` | `PopulationIR` with count N+1 |
| `new NetCon(src, tgt, thresh, delay, weight)` | `ProjectionIR` with weight and delay |

### Complete Example

See `configs/example_network.hoc` for a working example.

---

## Usage

### Loading Any Format

```cpp
#include "snnfw/declarative/DeclarativeLoader.h"

NeuralObjectFactory factory;
Datastore datastore("./db");
DeclarativeLoader loader(factory, datastore);

// Format is auto-detected from file extension
auto network = loader.loadNetwork("model.snnf.json");  // Native JSON
auto network = loader.loadNetwork("circuit_config.json"); // SONATA
auto network = loader.loadNetwork("model.nml");           // NeuroML
auto network = loader.loadNetwork("model.hoc");           // HOC
```

### Parse-Only Mode

```cpp
// Parse without constructing — useful for validation or format conversion
auto ir = loader.parseOnly("model.nml");
auto errors = ir.validate();
```

### Format Selection Guide

| Use Case | Recommended Format |
|----------|-------------------|
| New SNNFrame project | Native JSON (`.snnf.json`) |
| Importing from Allen Institute / Blue Brain | SONATA |
| Sharing with NeuroML ecosystem | NeuroML |
| Importing from NEURON simulator | HOC |
| Maximum SNNFrame feature access | Native JSON |
| Column templates + path-based connectivity | Native JSON |

---

## See Also

- [Developer Manual — Declarative Network Loading](DEVELOPER_MANUAL.md#declarative-network-loading)
- [Developer Guide — Declarative Loading Patterns](DEVELOPER_GUIDE_PATTERNS.md#declarative-loading-patterns)
- [Developer Guide — Custom Format Parsers](DEVELOPER_GUIDE_ADVANCED.md#custom-format-parsers)
- [API Reference — DeclarativeLoader](API_REFERENCE.md#declarativeloader)

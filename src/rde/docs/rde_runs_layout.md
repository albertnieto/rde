# RDE artifact layout

RDE writes durable artifacts below the caller's `store_root`:

```text
<store_root>/
├── campaigns/<campaign_id>/
│   ├── manifest.json
│   └── batches.jsonl
├── runs/<run_id>/
│   ├── manifest.json
│   ├── instances.jsonl
│   ├── instance_features.jsonl
│   ├── features.jsonl
│   ├── arrays/<instance_id>/*.npz
│   └── sealed/
│       ├── features.parquet
│       └── sealed.json
└── discovery/
    ├── <run_id>.json
    └── checkpoints/<run_id>/stages.json
```

JSONL is the rebuildable working ledger. NPZ files hold array-valued
primitives and slices when `save_arrays=True`. A sealed Parquet artifact is
written only after row-count and metadata verification. Discovery checkpoints
record completed stages, errors, and the population fingerprint.

`Store.flush()` and `Store.close()` make progress readable during a run.
Campaign manifests record requested and effective backends, resource limits,
configuration fingerprints, and stop reasons. Preserve the full store with
the experiment record; terminal output is not the experiment record.

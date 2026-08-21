# RDE engineering contracts

These contracts ship with the library and apply to core and plugin authors.

## Performance

- Choose the backend from `machine-profile`; keep computation and expression
  evaluation backend choices separate.
- Vectorize rows, assignments, populations, and target columns. Batch only
  compatible shapes and keep a scalar NumPy reference for parity.
- On MLX, cache device arrays, evaluate compatible populations in one graph,
  synchronize at deliberate boundaries, and avoid per-candidate host transfers.
- Reuse expensive domain caches. If `materialize` and `primitive_features`
  share work, implement `prepare_instance` and accept `cache=`.
- Bound chunks, workers, checkpoints, and retained artifacts. Unsupported
  descriptor families are excluded rather than padded.

## Domain and numerical contracts

- A domain's structural hooks expose raw input primitives and never consult a
  hidden answer or reference optimum.
- Metrics mask finite paired observations, guard denominators, reject
  constant/zero-variance inputs, and return `NaN` when a statistic is
  undefined.
- Preserve instance IDs, split assignments, targets, dtypes, shapes, and
  requested/effective backend provenance.
- Vectorization does not change brute-force asymptotics; keep explicit size
  guards and resource accounting.

## I/O, CLI, and observability

- Row dictionaries are an I/O boundary; materialize column arrays once.
- Propagate CLI options through discovery, reports, manifests, provenance, and
  resume fingerprints.
- Use JSONL/manifest/checkpoint artifacts for durable progress. Any run longer
  than a few seconds needs a genuine live work counter, elapsed time, and ETA.
  Core automatically attaches TTY progress or flushed newline progress for
  `run_pipeline` and `run_discovery` when callers do not provide a sink.
  `NullProgress()` is the explicit silence opt-out; size-only or
  end-of-stage output is not sufficient.
- Test real parser, plugin, storage, report, and subprocess boundaries.

The science contracts are in [methodology](methodology.md),
[architecture](ARCHITECTURE.md), and the
[experiment playbook](experiment-playbook.md).

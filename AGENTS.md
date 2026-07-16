# verl-vla Agent Guide

This file records the project decisions that coding agents must preserve when
changing verl-vla. Keep it focused on repository-specific constraints that
cannot be inferred reliably from the code alone.

## Project Direction

- verl-vla is a VLA post-training framework built on verl. It covers the
  connected lifecycle of human-in-the-loop data collection, supervised
  fine-tuning, reinforcement learning, and policy evaluation; these should
  remain parts of one system rather than separate model-specific pipelines.
- Extend verl for robotics and VLA workloads instead of copying or replacing
  its distributed training infrastructure. Reuse verl's `DataProto`, Ray/FSDP
  workers, resource management, and checkpoint mechanisms, and keep only
  VLA-specific behavior in this repository.
- Make new policies and simulators straightforward to integrate without forcing
  them into one implementation backend. Preserve upstream model code and native
  Hugging Face checkpoint formats; integration happens through explicit
  verl-vla adapters and builders.
- Treat documented examples as supported user workflows, not as internal
  experiments. Each maintained workflow should provide a reproducible Docker
  environment and a minimal launcher that takes the user from required inputs
  to a useful result.
- The project is still evolving. Prefer a clean, coherent architecture over
  compatibility with unused internal APIs, aliases, or configuration fields.
  Add a compatibility path only for a concrete, identified consumer.
- Do not turn verl-vla into an owner of upstream policies, simulators, robot
  runtimes, or datasets. Integrate their public contracts and pin a verified
  environment where needed; upstream-specific implementation remains upstream.

## Architecture Boundaries

- Keep the primary dependency direction one-way:
  `examples/docs -> entrypoints -> workflows -> trainers/TrainCluster -> workers -> models/envs`.
  Lower layers must not import entrypoints or workflows to trigger higher-level
  behavior.
- `examples/` and user documentation define the public launch experience. They
  may select configurations, prepare mounts, and invoke a CLI, but must not
  contain framework or algorithm implementations.
- `entrypoints/` is only the CLI-to-workflow boundary. An entrypoint selects the
  Hydra config and calls one workflow function; it does not construct models,
  workers, clusters, datasets, or trainers itself.
- `workflows/` owns end-to-end orchestration. It resolves workflow configuration,
  initializes Ray, composes multi-stage procedures such as ReCap, creates the
  appropriate `TrainCluster` and trainer, and guarantees cleanup. All composable
  Hydra YAML belongs under `workflows/config/`; trainers must not maintain a
  parallel configuration tree. The current PPO path predates `TrainCluster` and
  directly assembles workers and resource pools; treat it as a legacy exception,
  not as the pattern for new workflows.
- `trainer/` owns algorithm progression: optimization steps, replay or dataset
  iteration, validation cadence, metrics, and stopping behavior. A trainer uses
  the public `TrainCluster` operations and model training contracts; it does not
  decide Ray placement, instantiate simulator backends, or implement a user
  workflow.
- `train_cluster/` owns distributed topology and lifecycle. It maps actor,
  rollout, and environment roles onto Ray resources; starts and stops workers;
  coordinates rollout, training, evaluation, weight synchronization, and
  checkpoint lifecycle. It exposes these operations to trainers and workflows
  without containing model- or algorithm-specific policy logic.
- `workers/` owns execution inside distributed processes: device meshes,
  FSDP/optimizer execution, rollout RPCs, environment RPCs, and conversion at
  the `DataProto` transport boundary. Workers execute requests from
  `TrainCluster`; they do not choose workflow stages or user-facing defaults.
- `models/` owns model integration. Each integration keeps one upstream native
  policy inside a verl-vla TrainableModel wrapper, implements the training
  contracts it supports, and keeps model-specific adapter and auxiliary state
  outside the upstream policy. Model construction is explicit in
  `models/builder.py`; do not require Transformers AutoClass registration or
  convert a native checkpoint into a repository-specific format.
- Keep the two checkpoint purposes distinct. A full verl checkpoint contains
  the wrapper and training state needed to resume, while the Hugging Face export
  contains the native policy and its required processor/artifacts so it can be
  loaded by the upstream implementation without verl-vla.
- `envs/` owns simulator lifecycle and the normalized environment contract. A
  simulator backend translates its native API into the common observation and
  step schema; it must not depend on a trainer or a specific model. Translation
  between that schema and model-specific inputs/actions stays at the model
  policy/adapter boundary.
- `teleop/` and `recorder/` are independent domains integrated at the environment
  boundary. Device handling and intervention strategies stay in `teleop/`;
  dataset/video construction and recording strategies stay in `recorder/`.
  Workflow code may coordinate their outputs, but their implementation must not
  drift into `trainer/` or generic `utils/`.
- Use `utils/` only for small, stateless helpers shared by multiple domains. If
  code owns a lifecycle, configuration, backend choice, or domain data format,
  place it in the corresponding domain package instead.
- Keep typed configuration next to the component that consumes it, while Hydra
  composition and user-selectable defaults remain in `workflows/config/`.
  Model adapter configuration belongs to its model integration and must not be
  written into the upstream model's native config merely to transport runtime
  settings.

## Coding Conventions

- Treat tensors, `DataProto` objects, configuration objects, and similar
  structured data as explicit contracts. Use one canonical access and parsing
  path; do not add defensive fallback chains, repeated shape/type guesses,
  permissive defaults, or multiple `if`/`getattr` branches to accommodate
  hypothetical inputs. When a contract is unclear—for example, whether an
  action tensor is two- or three-dimensional—trace the value back to its
  producer and establish the correct shape at the source instead of accepting
  both shapes at the consumer. Conditional parsing is allowed only when the
  supported data sources and their distinct schemas are explicitly identified.
- Treat the `workflows/config/` YAML tree as the public configuration
  architecture and make its composition mirror runtime ownership. Each file
  should define one reusable component, workflow stage, or selectable variant;
  parent configs assemble owned children through Hydra defaults/package paths,
  while local YAML contains only fields and intentional overrides owned at that
  layer. Reuse base configs and config groups instead of copying configuration
  blocks, preserve the resulting ownership hierarchy in user override paths,
  and do not flatten child domains or expose parallel aliases for the
  convenience of an implementation. When an upstream API expects another
  schema, adapt the canonical composed config once at the typed consumer
  boundary. Keep `_self_` last when local values are intended to override
  composed defaults.
- When designing a user-facing launch workflow, follow the orchestration style
  of the Pi0.5 SFT guide: provide one minimal entrypoint, expose only inputs the
  user must choose, and keep internal setup and composition behind that
  boundary. A user should be able to reproduce the documented workflow
  end-to-end with as few commands as practical.
- Add tests only for necessary core behavior that forms a complete,
  independently meaningful unit and contains enough logic to warrant dedicated
  verification. Typical examples include the correctness of an algorithm or
  formula, computed data statistics, and execution or scheduling state
  transitions. Do not add tests for every private helper, internal call
  sequence, trivial forwarding layer, configuration default, or hypothetical
  defensive branch. Test the stable behavior at its owning boundary without
  freezing incidental implementation details.

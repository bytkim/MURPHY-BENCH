MURPHY
======

MURPHY compares the token and probability records returned by two llama.cpp
builds. It keeps instruction-following compliance separate from record
integrity, and reports unavailable measurements separately from passes.

Dataset, report and related PR
------------------------------

* `Dataset and evidence <https://huggingface.co/datasets/bytkim/MURPHY-BENCHMARK/tree/c930bbd09898797c3a2b8206859866d596be56f9>`_: the exact 1,500 evaluated tasks, build manifests, raw evaluation results and bisect evidence.
* `Evaluation report <https://huggingface.co/datasets/bytkim/MURPHY-BENCHMARK/blob/c930bbd09898797c3a2b8206859866d596be56f9/evidence/evaluation-report.rst>`_: baseline versus patched results, separated by dataset, endpoint and mode.
* `llama.cpp PR #31244 <https://github.com/ggml-org/llama.cpp/pull/31244>`_: the related token-ID and probability-record buffering fixes.

The 24,000-request evaluation used baseline ``8a199ad31a7e`` and patched
``636d96d2e686``, not the current two-commit PR stack. See
`evaluation provenance <docs/provenance.rst>`_ for full source and model pins.

Quick start (no server required)
-------------------------------

Use Python 3.11 or newer from this checkout::

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements-dev.txt
    .venv/bin/python -m pytest -q
    .venv/bin/python examples/mock_evaluation.py --output data/dev/mock-example

The example uses synthetic fixtures and mocked HTTP, not a model or a live
server. Its deliberately defective baseline demonstrates the report; it is
not qualification of an inference runtime. Choose a fresh output directory
for each invocation. Generated data is excluded from version control.

System
------

* ``adapters/`` converts source datasets into fixed tasks and manifests.
* ``murphy/contract/`` defines versioned input and output schemas.
* ``murphy/client.py`` captures HTTP and SSE without repairing server output.
* ``murphy/core/`` runs, grades, measures and compares saved responses.
* ``murphy/util/vocabulary.py`` reads the GGUF vocabulary for byte-label references.
* ``scripts/evaluate.py`` admits and runs two explicitly configured local
  servers; ``scripts/report.py`` produces endpoint-separated reports offline.

Outputs follow ``data/<dataset>/<build>/`` with ``manifest.json``,
``tasks.jsonl``, ``runs/<target>/`` and ``compare/``. Comparisons require the
same task build. Raw chunks and terminal records remain evidence rather than
being flattened into a repaired response.

Paired-endpoint policy
----------------------

Select ``policies/murphy.v4.json`` explicitly: historical defaults are retained
for reproducibility. The full profile contains 900 FLORES+, 300 Belebele and
300 ten-entry emoji prompts. Two endpoints, two modes, two repeats and two
builds yield 24,000 requests. See ``docs/data.rst`` before preparing inputs::

    python scripts/prepare.py --sources SOURCES --model-metadata MODEL_METADATA --output data/dev/prepared --policy policies/murphy.v4.json --profile full
    python scripts/prepare_datasets.py --build PREPARED_BUILD_PATH --output data

Use the build directory printed by the first command as ``PREPARED_BUILD_PATH``.

An experiment recipe binds manifests and target TOML files by SHA-256, plus
the template where applicable. Targets pin model and binary hashes, source
revision, explicit runtime settings and loopback server addresses. Validate
an existing recipe without launching servers::

    python scripts/evaluate.py experiment.json --check

Actual execution is a separate operation::

    python scripts/evaluate.py experiment.json --output data/dev/experiment

Inspect ``murphy/contract/experiment/experiment-v1.schema.json`` and
``murphy/contract/core/target/target-v1.schema.json`` for
the exact contract. This source package does not include model weights,
prebuilt servers, dataset snapshots or a cloud provisioning service.

Interpretation
--------------

Reports are organized by dataset, endpoint, mode and target build. Response
completion and task compliance are separate from missing records, wrong
token identities, byte-label mismatches and numerical disagreement.

Native completion can supply positional token references. Chat accounting
uses whole-response ``usage.completion_tokens`` where available; equal counts
do not establish identities or ordering. Chat positional coverage,
actual-output exposure and numerical agreement remain unavailable under the
current eligibility rules. Byte checks require suitable records and the
correct model vocabulary. Numerical agreement between builds is not accuracy
against an independent reference, nor parity between different endpoints.

Licensing and provenance
------------------------

MURPHY source is licensed under Apache-2.0; see ``LICENSE``. Dataset and model
licenses are separate. See ``docs/data.rst`` and ``docs/provenance.rst``.

Evaluation attribution
======================

The original 24,000-request paired-endpoint evaluation used MURPHY source
``9557ccdfd71acd2bd47d125b24ca43e559b49f04`` and these llama.cpp revisions:

* Baseline: ``8a199ad31a7eb27d5a46a6b08026176d82bc9894``.
* Patched: ``636d96d2e686690011efc1f6a265692c380ab44b``.

The model was Qwen3.5-9B BF16, GGUF SHA-256
``30a6732abca868d08fe0e9abc9a69d0b2c8653cd9ad07e60e8ea675fef06a1be``.
Flash attention and backend sampling were off; context 4096, batch and
microbatch 512, GPU layers 99, one thread and slot, temperature 0, seed 42,
five probabilities and maximum 384 generated tokens. Thinking was disabled;
no literal grammar constrained the evaluation.

Do not attribute those results to the proposed two-commit submission stack
ending at ``d8317e0e7f20998058cd4c87a56ca5b86f7db88a``, or to a subsequent
upstream port. Although the combined fix has the same stable patch identity,
the surrounding source trees and runtime identities differ. Local regression
tests on a newer upstream base are separate evidence, not a rerun of that
evaluation. Packaging additions do not change the 61 evaluated source files.

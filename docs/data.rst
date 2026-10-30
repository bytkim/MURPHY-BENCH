Dataset inputs and attribution
==============================

Source snapshots must be acquired separately under their original terms.
MURPHY does not bundle or automatically download them. Apache-2.0 covers
MURPHY source, not these datasets or the target model.

* `FLORES+ <https://huggingface.co/datasets/openlanguagedata/flores_plus>`_,
  maintained by the Open Language Data Initiative, derives from FLORES-200.
  Dataset license: CC BY-SA 4.0. Policy v4 pins revision
  ``1147e2dc59f37fbb5c22fd8478770de02cd399f9`` and the dev split.
* `Belebele <https://github.com/facebookresearch/belebele>`_, Meta's multilingual
  reading-comprehension benchmark. Dataset license: CC BY-SA 4.0. Policy v4
  pins revision ``7899cdfa4e1e0d733fd77c848e2c273cb1d32be2``.
* `Unicode Emoji 17.0 <https://www.unicode.org/emoji/charts-17.0/>`_, Unicode,
  Inc. Retain the applicable `Unicode license <https://www.unicode.org/license.txt>`_
  and notices with downloaded data. Policy v4 uses emoji-test.txt version 17.0.0.

The source root contains ``preservation/source-manifest.json``, whose ``files``
array records each relative ``path``, byte count (``bytes``), and ``sha256``.
Adapters verify files against that inventory before use. Required paths are::

    sources/flores_plus/<revision>/dev/<language>.jsonl
    sources/belebele/<revision>/data/<language>.jsonl
    sources/unicode_emoji/17.0.0/emoji/emoji-test.txt

FLORES rows retain ``id`` and ``text``. Belebele rows retain ``link``,
``question_number``, ``flores_passage``, ``question``, ``mc_answer1`` through
``mc_answer4`` and ``correct_answer_num``. Preserve aligned IDs rather than
resampling each language independently. The adapters are the authoritative
input-format definitions.

Language inputs are eng_Latn, rus_Cyrl, spa_Latn, arb_Arab, jpn_Jpan,
zho_Hans and zho_Hant. Tokenizer probes select one Chinese script, recording
split rates in the manifest; ties choose Simplified. Model metadata must
match the policy's pinned revision, including tokenizer and chat template.

The full policy selects 150 aligned FLORES sentences across six languages
for exact repetition, and 50 aligned Belebele questions across six languages
for answer-letter plus original choice text. Emoji uses 60 unique sequences
per family and 300 prompts of ten entries each, two entries from each family:
single code point, modifier without ZWJ, ZWJ, flags including tag flags, and
other multi-code-point sequences (keycaps and variation-selector sequences).
Each entry requests the emoji and its supplied English name in order.

Prompt templates and policy versions are hashed inputs. Altering wording,
grouping, selection or endpoint conditions creates a new build; it must not
overwrite an earlier evaluation.

# Pilot 04: authored meaning transfer and counterfactual rejection challenge

**Status:** frozen proposal and untested challenge before the first CI run. Preserve raw outcomes regardless of performance.

## Question and purpose

Pilot 03 improved lexical perturbation identification in a synthetic masked Hebbian model by changing cue encoding, without enlarging the weight matrix. That did not prove semantic recollection. Pilot 04 is a deliberately harder, small **assistant-authored, non-blind** challenge to distinguish lexical association from actual use of narrative meaning and contradictions.

The target source remains the v12 autobiography: 450 reconstructed fictional memories, 27 episodes, with exact frozen source hashes enforced by the runner. Pilot 04 also pins its *new* 68-case challenge file as a Git blob. Neither source nor challenge is edited after testing. No FlyWire biological topology, free-form textual memory, or agent behavior is implemented.

## Challenge content and provenance

`experiments/pilot04/challenge_v1.jsonl` contains **68 human-readable prompts across 34 event IDs and 23 episodes**. For each source event, one **meaning-preserving paraphrase** describes an incident or its consequence in new language, and one **contradiction** asserts a fact inconsistent with the autobiographical record. The source anchor provides the author rationale, but it remains evaluation metadata outside the model. There is no generated-train text, and no prompt is introduced into original recall cues.

**Authoring caveat:** all 68 prompts were **written by ChatGPT after viewing relevant source text**; they have **not** been validated by an independent human annotator, and the author was not blinded. They cannot support a publication-grade claim of human-validated semantic generalization. They provide a frozen exploratory stress set, not an independent benchmark. Some questions may still share words with existing cues, and paraphrase validity is provisional until editorial review.

## Key distinction: three negative and positive conditions

**Learned-event true paraphrases:** if an event lies in training episodes, the corresponding novel phrase asks whether the network can identify it using the *external oracle fingerprint codebook*. Report top-1, acceptance rate and correct-and-accepted rate.

**Learned-event contradictions:** same trained event, but the text describes a false or counterfactual outcome. Correct behavior is **abstention** rather than retrieving the closest real record. Report any false acceptance, and especially *false confirmation of the contradicted event*. These cases are NOT categorized as unknown-event negatives in `scores()` because their event ID genuinely is in the candidate codebook; their logical truth label comes only from the held-out challenge metadata.

**Unlearned-event true paraphrases:** if source event belongs to the held-out test episodes, the underlying record was never imprinted, so the model should abstain. Report false acceptance. Validation episodes, test episodes and trained episodes never overlap within a seed.

## Frozen models and calibration

Reuse Pilot 03's seven methods **unchanged**: legacy-full, legacy-surface, token, char3, train-only IDF-char3, token-shuffled targets and untouched network. Keep the 512-cue by 256-fingerprint synthetic mask, 0.55 density and original Hebbian plasticity, using seeds `31,37,43` to compare the exact trained configurations from Pilot 03. The model never sees its target event ID or its positive/negative label: every challenge query receives the **same neutral query ID**. Decoder text and true labels remain outside neural weights.

Per model/seed, calibrate the acceptance threshold using **only original Pilot 03 lexical swapped-cue positives from training events and separate validation-episode negatives**. Do not tune on authored paraphrases or contradictions. The threshold is frozen and applied to all challenge types.

Record exact per-case predicted IDs, accepted/rejected states, top cosine and source episode groups in the full JSON result artifact. Unlike closed-set accuracy, an all-accept system must be penalized for both contradictory and absent-event cases. The shuffled mapping control is especially important because Pilot 03 found it can detect familiar input statistics without identifying memories.

## Non-neural reference

Include Pilot 03's conventional word-overlap lookup for true paraphrases. For contradictions, report how often that forced-choice lookup outputs the contradicted event ID (an apparent false confirmation); it cannot abstain. This comparison is illustrative rather than resource-matched.

## Reproduce and quality gate

```sh
python -m pip install "numpy>=1.26,<3" "pyarrow>=17,<24"
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 python scripts/run_imprinting_pilot04.py --seeds 31,37,43 --output results/imprinting/pilot04.json
```

The dedicated Pilot 04 workflow validates both memory and challenge Git blobs, runs the full regression suite, executes the three-seed assay, and uploads full case-level JSON. The results report must be written **after** observing the actual CI output, not guessed.

## Limits and decision gates

The model does not read or semantically embed the memory narrative at imprint time: it is given only cue strings and an opaque numeric target derived from the narrative. Therefore, a poor performance on semantically reworded story facts is not surprising, and is a meaningful architectural limit. Neither a good paraphrase result nor an accurate rejection score demonstrates generated memories, stable values, or decision-making.

A syntactically explicit contradiction may share many words with authentic cues. The current encoder has no truth-verification logic, so a high contradictory false-acceptance rate is a serious anticipated failure, not a surprising implementation defect.

If the semantic challenge performs poorly, the next controlled branch should compare the **same** synthetic overlay with a source-grounded semantic representation and an explicit contradiction/entailment verification layer. Freeze fresh, independently reviewed prompt labels before such tuning. Do not report the original Pilot 04 cases as new untouched test data after reading their outcomes. Only after a rigorous memory retrieval and truth-checking gate should behavioral persona claims or biological FlyWire overlay scaling be considered.

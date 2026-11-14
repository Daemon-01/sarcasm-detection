# AGENTS.md

## 1. Project Mission

You are working on **SarcasTone: Multimodal Sarcasm Detection Using Text and Speech Prosody**.

The immediate implementation scope is the **MUSTARD++ text branch only**.

The broader project intends to compare:

- Text-only
- Speech-only
- Multimodal text + speech

However, implement only the text branch at this stage. Maintain minimal interfaces for future speech and fusion integration.

**Current timeline:** Approximately 2–3 weeks.

Prioritize reproducibility, experimental validity, clean architecture and defensible results over unnecessary infrastructure.

## 2. Source of Truth: OKF

Before making any changes:

1. Read `docs/sarcastone_text_branch_okf/index.md`.
2. Follow the relevant concept pages.
3. Read `docs/sarcastone_text_branch_okf/log.md` for historical decisions.
4. Inspect the existing repository before creating or modifying files.

The **OKF is the project's specification and source of truth**.

This file defines agent behaviour and implementation rules. The OKF contains the detailed project knowledge and decisions.

If existing code conflicts with the OKF:

- Do not silently choose an interpretation.
- Prefer the latest explicitly dated decision.
- Record unresolved conflicts.
- Ask for clarification when necessary.
- Never change a documented decision simply for implementation convenience.

Never invent dataset facts, model details, experimental results or research claims.

Use `TODO` for unresolved requirements.

## 3. OKF Documentation Conventions

The OKF must remain a Markdown-based knowledge directory.

### Required conventions

- One focused concept or decision per Markdown file.
- Every concept file must contain YAML frontmatter.
- The `type` field is mandatory.
- Use relative Markdown links.
- Maintain `docs/sarcastone_text_branch_okf/index.md` as the main entry point.
- Maintain `docs/sarcastone_text_branch_okf/log.md` as the dated decision history.
- Keep concept files concise and focused.
- Update the index whenever a new concept file is added.
- Record important decisions and changes in the dated log.

### Frontmatter format

```
---
type: concept
---
```

Use an appropriate `type` value for each document.

### Decision log format

```
## YYYY-MM-DD

- **Decision:** ...
- **Reason:** ...
- **Related:** [Concept](relative-path.md)
```

Never rewrite historical decisions. Add a new dated entry when something changes.

Do not introduce knowledge-management SDKs, extra services or unnecessary tooling.
## 3.1 Unified Multimodal OKF and Interface Standards

The existing OKF directory structure is authoritative. Preserve its current organization and extend it only when necessary.

The OKF must support documentation for:
- Text-only experiments.
- Speech-only experiments.
- Multimodal fusion experiments.

All branches must follow the same Markdown, YAML frontmatter, navigation and decision-log conventions.

Do not create duplicate documentation directories or reorganize existing knowledge unnecessarily.

### Shared feature interface

Text and speech extractors may use different algorithms, libraries and feature dimensions.

However, they must follow a common input/output contract:

1. Accept the standardized manifest and requested utterance IDs.
2. Return modality-specific feature representations.
3. Preserve the association between each representation and its utterance ID.
4. Use shared feature storage and loading utilities.
5. Preserve sample ordering and relevant metadata.
6. Report missing or invalid samples explicitly.

Sparse and dense representations must both be supported where required.

The fusion component must be able to align text and audio features using their shared utterance IDs.

### Implementation scope

The current implementation scope remains the text branch.

Speech and fusion documentation may describe their intended interfaces and research requirements, but their actual processing, models and training must remain unimplemented until explicitly authorized.

Do not introduce speculative implementations or unnecessary abstractions.

## 4. Dataset and Preprocessing

### Dataset restrictions

- Use MUSTARD++ only.
- Do not mix in MUSTARD or other datasets.
- Implement only the text branch.

### Target and context identification

Use the `Sarcasm` column.

- Target utterances: `Sarcasm` is non-null.
- Context utterances: `Sarcasm` is null.
- Utterance text: `SENTENCE`.
- Conversation grouping: `SCENE`.

A null context label does not mean a non-sarcastic target.

Context utterances provide conversational information. They are not independent target examples.

### Context ordering

Within each `SCENE`:

1. Extract the numeric suffix from `KEY`.
2. Sort context utterances in ascending numeric order.
3. Preserve gaps in the sequence.
4. Never fabricate missing turns.
5. Never renumber original keys.

Do not sort KEY values as ordinary strings.

### Text preservation

Preserve:

- Original wording.
- Capitalization.
- Punctuation.

Only normalize whitespace and line breaks when necessary.

Do not apply:

- Forced lowercasing.
- Stop-word removal.
- Stemming.
- Lemmatization.
- Unapproved text rewriting.

Keep preprocessing deterministic and documented.

## 5. Context Representation

The project uses two representation stages.

### Option A: Structured representation

Represent each example using structured fields:

- SCENE identifier.
- Target utterance.
- Ordered context-turn list.
- Relevant turn metadata.
- Original keys where required.

Keep target and context as separate logical fields.

Each context turn must remain individually recoverable.

### Option B: Serialized representation

Convert the structured representation into model-ready text.

Requirements:

- Use `[TURN]` as the agreed context-turn separator.
- Preserve the original ordering of context turns.
- Keep the target utterance logically separate from the context.
- Do not merge target and context into an indistinguishable text field.
- Preserve the original structured representation.

Do not assume `[TURN]` is automatically a tokenizer-native special token.

**Tokenizer Handling (Resolved 2026-11-14):** `[TURN]` is treated as plain text by the RoBERTa-base tokenizer (it is **not** registered as a special token). The tokenizer splits it into sub-word tokens (e.g., `[`, `TUR`, `##N`, `]`). This is the confirmed and accepted behaviour: since RoBERTa-base remains frozen with attention-mask-aware mean pooling, `[TURN]` subwords still contribute to the pooled representation as regular tokens. No special-token registration is required for the current frozen encoder setup.

## 6. Text Model Architecture

The current model direction is:

- Encoder: RoBERTa-base.
- Encoder: Frozen.
- Pooling: Attention-mask-aware mean pooling.
- Classifier: Small sarcasm classification head.

### Pooling requirements

Padding tokens must not contribute to the pooled representation.

Use the attention mask when calculating the mean-pooled embedding.

Keep the encoder frozen during the current experiments.

Do not introduce encoder fine-tuning, larger architectures or additional trainable modules without an explicit project decision.

### Baseline

Use TF-IDF + Logistic Regression as a comparison baseline.

Important distinctions:

- This is a separate baseline experiment.
- It is not a substitute for evaluating RoBERTa.
- Maintain consistent evaluation splits wherever applicable.
- Report the baseline and neural model independently.

### Unresolved model details

Keep the following as TODO until confirmed:

- Exact sentiment checkpoint, if applicable.
- Exact model export/checkpoint format.
- Tokenizer-specific `[TURN]` handling.
- Any additional unspecified model components.

Never invent checkpoint names or export conventions.

## 7. Experimental Protocol

All relevant comparisons must use a shared:

- Speaker-independent split.
- Frozen 5-fold cross-validation assignment.
- Three-seed experimental protocol.

### Requirements

1. Establish speaker-independent folds.
2. Generate one shared fold assignment.
3. Reuse the same folds across relevant model comparisons.
4. Freeze the fold assignment after generation.
5. Run three seeds for the experiments.
6. Record the seed values.
7. Record fold assignments or their reproducible generation procedure.
8. Make fold-level and seed-level variation inspectable.

Do not generate different folds for different models merely for convenience.

Use frozen encoders and small trainable heads in accordance with the current experimental direction.

### Metrics

Use the metrics explicitly specified by the project owner or OKF.

Record:

- Metric definitions.
- Averaging conventions.
- Fold-level results.
- Seed-level results.
- Aggregated results.

**TODO:** Confirm the final metric list and averaging conventions if they are not documented.

Do not silently invent evaluation requirements.

### Valence and arousal

Valence/arousal information is **validation-only**.

Do not use it as:

- A training target.
- A test-time input.
- A source of test-set tuning.

Preserve this restriction unless a later documented decision changes it.

## 8. Speech and Multimodal Branches

The current implementation is text-only.

Only one minimal interface stub for each future component is permitted:

- Speech branch interface.
- Text/speech fusion interface.

Do not implement:

- Audio processing.
- Speech feature extraction.
- Speech model training.
- Multimodal fusion logic.
- Additional audio dependencies.
- Fabricated placeholder predictions.

Clearly mark both components as not implemented.

Avoid speculative abstractions.

## 9. Engineering Rules

- Inspect existing repository conventions before editing.
- Prefer readable, modular and maintainable code.
- Separate preprocessing, splitting, model definition, training, evaluation and reporting.
- Make transformations deterministic.
- Record random seeds.
- Prevent data leakage.
- Fit preprocessing components only on the appropriate training data within each fold.
- Never use test information for model fitting or tuning.
- Do not silently drop examples.
- Log exclusions and their reasons.
- Validate labels, grouping, ordering and speaker separation.
- Keep credentials and access tokens out of source files and logs.
- Avoid unnecessary dependencies and external services.
- Do not introduce features merely because they might be useful in future branches.

## 10. Experiment Traceability

Every experiment must record:

- Dataset and version/source.
- Preprocessing specification.
- Model architecture.
- Frozen and trainable components.
- Fold assignment.
- Speaker-independence policy.
- Random seed.
- Training configuration.
- Hyperparameters.
- Validation procedure.
- Metrics and averaging conventions.
- Final evaluation outputs.
- Failed runs and deviations.

Every reported result must be traceable to its configuration.

Never fabricate missing experimental values.

## 11. Handling Unknowns and Changes

When a requirement is unclear:

1. Check the relevant OKF page.
2. Check the dated decision log.
3. Inspect existing code and artifacts for supporting evidence.
4. Identify the unresolved question.
5. Mark it as TODO.
6. Ask the project owner before making consequential assumptions.

When a documented decision changes:

1. Explain the reason.
2. Update the relevant concept page.
3. Add a dated entry to `docs/sarcastone_text_branch_okf/log.md`.
4. Update `docs/sarcastone_text_branch_okf/index.md` if necessary.
5. Preserve the historical decision.

## 12. Completion Checklist

Before declaring a task complete, verify:

- [ ] Latest OKF decisions have been followed.
- [ ] Only MUSTARD++ is used.
- [ ] Target and context remain separate.
- [ ] SCENE grouping is correct.
- [ ] KEY suffixes are sorted numerically.
- [ ] Text preservation rules are respected.
- [ ] Option A remains recoverable from Option B.
- [ ] `[TURN]` tokenizer behaviour is documented.
- [ ] RoBERTa-base remains frozen.
- [ ] Mean pooling uses the attention mask.
- [ ] Shared speaker-independent 5-fold CV is used.
- [ ] Three seeds are accounted for.
- [ ] Valence/arousal remains validation-only.
- [ ] Speech and fusion remain interface stubs.
- [ ] No undocumented dependencies or experimental assumptions were introduced.
- [ ] Relevant OKF pages and dated logs are updated.

### Final task report

At the end of every implementation task, report:

1. What changed.
2. Which files changed.
3. Which checks and experiments were performed.
4. Actual measured results.
5. Remaining TODOs.
6. Limitations or deviations from the OKF.

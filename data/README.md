# Dataset — Prompt Injection Detector

## Format

Each file uses [JSON Lines](https://jsonlines.org/) format (`.jsonl`): one JSON object per line.

### `injections.jsonl`

Malicious prompt examples labeled as injection attempts.

```json
{"text": "Ignore all previous instructions...", "label": 1, "category": "instruction_override"}
```

**Fields:**
- `text` (string): The prompt text
- `label` (int): Always `1` (injection)
- `category` (string): Attack category — one of:
  - `instruction_override` — Attempts to override system instructions
  - `jailbreak` — Known jailbreak techniques (DAN, AIM, etc.)
  - `role_hijacking` — Forces the model into a different role
  - `prompt_extraction` — Tries to extract system prompt
  - `delimiter_injection` — Uses markup tags to inject instructions
  - `encoding_trick` — Obfuscates injection keywords
  - `context_manipulation` — Uses hypothetical framing or authority claims
  - `output_manipulation` — Forces specific output format to bypass filters

### `legitimate.jsonl`

Legitimate prompt examples (safe inputs).

```json
{"text": "What is the capital of France?", "label": 0}
```

**Fields:**
- `text` (string): The prompt text
- `label` (int): Always `0` (legitimate)

## Sources

All examples in this dataset are **synthetically generated** based on publicly documented prompt injection techniques. No real user data is included.

Techniques documented in:
- [OWASP LLM Top 10](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- Greshake et al. (2023) — "Prompt Injection Attacks against LLM-integrated Applications"
- [PromptInject](https://github.com/agencyenterprise/promptinject) (reference dataset)
- Public jailbreak documentation for research purposes

## Statistics

| File | Samples | Labels |
|------|---------|--------|
| `injections.jsonl` | ~50 | `1` (injection) |
| `legitimate.jsonl` | ~50 | `0` (legitimate) |

## Disclaimer

This dataset is for **security research and educational purposes only**.
The injection examples demonstrate known attack patterns to help build
better detection systems. They should not be used to attack real systems.

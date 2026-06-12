# Baseline Eval: Why ~51% Accuracy with ~6% Format Score

## Result (2026-06-11, `--no-restore --preset greedy`)

| Metric | Score |
|---|---|
| Accuracy | 51.56% (33/64) |
| Partial accuracy | 53.12% |
| Format accuracy | 6.25% |

## The apparent paradox

The base Gemma-3-1B-it model solves ~half the GSM8K problems correctly while almost never
producing the expected XML template. These two facts are consistent because accuracy and format
are measured by **different regexes**.

## How the scoring works

`rewards.py` defines two patterns:

```python
# Full template required: <reasoning>…</reasoning><answer>…</answer>
match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

# Just needs <answer> followed by any digit, anywhere in the response
match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)
```

`evaluate.py` uses `match_numbers` for the accuracy check:

```python
ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else "-1e9"
if float(ext.strip()) == float(ans.strip()):
    got_corr = True
```

and `match_format` only for the format check:

```python
if match_format.search(r) is not None:
    got_format = True
```

## What the base model is actually doing

Gemma-3-1B-it is instruction-tuned and good at grade-school math. A typical completion looks
something like:

```
Here is my step-by-step solution...

<answer>42
```

- No `<reasoning>` / `</reasoning>` tags → `match_format` fails → format score = 0
- `<answer>42` is present → `match_numbers` extracts `42` → accuracy credited if correct

## What this means for RL training

The base model has latent arithmetic ability; it is just not using the required schema. RL
training (GRPO with `match_format_exactly` and `match_format_approximately` rewards) is
specifically designed to teach the format. The expected training trajectory is:

- **Format score** climbs toward ~100% as the model learns to emit the full
  `<reasoning>…</reasoning><answer>…</answer>` template.
- **Accuracy** should stay roughly flat or improve as the structured reasoning chain
  becomes more reliable.

A large gap between accuracy and format score at the start of training is therefore **expected
and healthy** — it means there is a clear shaping signal for the format rewards to exploit
without needing to teach the underlying math from scratch.

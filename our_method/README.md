# Boolean Genetics: Inference-Time Sampling

This project compares fixed-budget Boolean-expression proposals with and
without sampling feedback. Both methods make the same number of model calls per
task: by default, the number of mechanistically distinct ground truths, capped
only when `--max-attempts` is supplied. The sampler records every proposal,
including malformed outputs, so VR, NR, and RR use explicit and comparable
denominators.

## Setup

Run the following commands from the `our_method` directory:

```powershell
Set-Location .\our_method
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The official dataset generator requires NumPy and SymPy:

```powershell
python ..\_HypoSpace-main\boolean\boolean_dataset.py --operators extended --max-depth 1 --seed 42 --output data\boolean_dataset.json
```

AINative can provision a temporary API key on the first real request; no
account setup is needed. The SDK saves that key in
`~/.ainative/config.json` and prints a claim URL for permanent access. Prompts
and benchmark observations are sent to AINative's API:

```powershell
$env:AINATIVE_MODEL = "gpt-oss-120b"
```

The default model is `gpt-oss-120b`, which was verified with the automatically
provisioned Basic-tier key. The suggested
`meta-llama/Llama-3.3-70B-Instruct` currently returns HTTP 403 on that tier;
it requires a plan upgrade. Set `AINATIVE_MODEL` to override the default. The
SDK retries transient API errors up to three times.

For an OpenAI-compatible endpoint such as OVHcloud's, configure the endpoint
and model:

```powershell
$env:LLM_BASE_URL = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
$env:AINATIVE_MODEL = "gpt-oss-120b"
python baseline.py --samples 1 --seed 42
python evaluate_ours.py --samples 1 --seed 42 --max-attempts 3
```

When `LLM_BASE_URL` is set, the client uses `LLM_API_KEY` if provided, or the
`not-needed` placeholder by default; it does not send an AINative key to the
custom endpoint. For OVHcloud's anonymous host, the client explicitly omits
the `Authorization` header. Anonymous access depends on the provider's current
policy; a 403 response may mean that endpoint access requires authentication.
For Hugging Face, set `LLM_BASE_URL` to
`https://router.huggingface.co/v1`, `LLM_MODEL` to a provider-supported model
ID, and `LLM_API_KEY` in your local environment.

For OpenRouter, set your API key as an environment variable. The adapter will
automatically use OpenRouter's OpenAI-compatible endpoint and the `openrouter/free`
model router; set `LLM_MODEL` if you want to choose a specific model instead:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
$env:LLM_MODEL = "openrouter/free"
python baseline.py --samples 1 --seed 42
```

## Run

### Offline mock (no API key or network request)

The deterministic mock intentionally returns malformed text, an inconsistent
Markdown-fenced expression, a labeled answer, and a duplicate before returning
valid candidates. This exercises the parser, rejection loop, CSV writing, and
plotting pipeline. These outputs are synthetic and are **not experimental
evidence**:

```powershell
python baseline.py --mock --samples 3 --seed 42
python evaluate_ours.py --mock --samples 3 --seed 42
python compare_results.py --baseline results\baseline_mock.csv --ours results\ours_mechanistic_mock.csv --summary results\summary_mock.csv --chart results\recovery_mock.png
```

![Mock-only pipeline demonstration; not real model results](results/recovery_mock.png)

### Real model

Run both experiments with the same model, seed, and dataset so their sampled
tasks and per-task proposal budgets match. The baseline uses independent
proposals without history or duplicate rejection; the default sampler uses
mechanistic deduplication and feeds accepted hypotheses/rejection feedback into
later prompts:

```powershell
python baseline.py --samples 10 --seed 42
python evaluate_ours.py --samples 10 --seed 42
python compare_results.py --baseline results\baseline.csv --ours results\ours_mechanistic.csv
```

If either run is interrupted, rerun the same command with the same
`--samples`, `--seed`, and `--output`, adding `--resume` to skip tasks already
saved in that CSV. Each completed task is flushed and synced to disk before the
next one begins.

Outputs are written under `results\`. The report includes:

- **VR**: proposals consistent with the observations and hypothesis-space
  constraints, divided by the number of model calls.
- **NR**: unique in-space mechanistic proposals, whether or not they match the
  observations, divided by the number of model calls.
- **RR**: recovered mechanistic ground truths divided by the enumerated
  mechanistic ground-truth set.
- **Simple/Complex RR**: RR split by operator count (`c <= 3` and `c > 3`).
  A stratum with no ground-truth hypotheses is reported as `n/a`.

The semantic recovery metric remains available as a secondary diagnostic.
Each CSV row also stores the ordered extracted candidates and raw model
responses as JSON arrays, including malformed responses, for failure analysis.
`evaluate_ours.py --dedup` accepts `none`, `semantic`, or `mechanistic`;
`--no-feedback` disables accepted-hypothesis and rejection feedback while
retaining the selected deduplication behavior. For example, compare the
deduplication modes under feedback using separate outputs:

```powershell
python evaluate_ours.py --samples 10 --seed 42 --dedup none --output results\ours_none.csv
python evaluate_ours.py --samples 10 --seed 42 --dedup semantic --output results\ours_semantic.csv
python evaluate_ours.py --samples 10 --seed 42 --dedup mechanistic --output results\ours_mechanistic.csv
```

Each completed task is appended to its CSV immediately. Existing result files
with older column layouts are rejected rather than silently mixed with the new
metrics; choose a fresh `--output` path when upgrading.

The checked-in dataset is a small, low-depth Boolean setting and does not
contain `c > 3` ground truths. Its Complex RR is therefore `n/a`; use the
paper-matched hard dataset before drawing conclusions about complex-hypothesis
coverage. The local mock only exercises pipeline behavior and is not
experimental evidence.

For AINative, the adapter reads `AINATIVE_MODEL`; set `AINATIVE_API_KEY` to
avoid automatic provisioning. For OpenRouter, `OPENROUTER_API_KEY` selects the
provider and `LLM_MODEL` overrides its default model.

## Limitations

- The fixed proposal budget does not guarantee a target number of valid,
  unique hypotheses; invalid and repeated outputs consume budget.
- Output extraction handles common labels and Markdown fences, but other
  malformed or novel response formats may still be rejected.
- Truth-table semantic deduplication is exponential and supports at most 16
  variables.
- Mock mode only validates software behavior; only real model runs can support
  conclusions about hypothesis recovery.

Run the local tests with `python -m unittest discover`. They cover expression
validation, semantic/mechanistic deduplication, fixed proposal budgets,
malformed mock outputs, API retry handling, and VR/NR/RR reporting.

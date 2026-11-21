# General-Tool-Agent

This project implements the NeurIPS 2024 GTA-1 benchmark:
[GTA: A Benchmark for General Tool Agents](https://arxiv.org/abs/2407.08713)
and the [open-compass/GTA repository](https://github.com/open-compass/GTA).

## Benchmark implemented

GTA-1 contains 229 human-written real-world tasks, 252 images, 14 tools, and
557 executable ground-truth tool calls.

The implementation provides:

- Official GTA-1 `dataset.json` and `toolmeta.json`
- Ground-truth tool-chain extraction
- Step-by-step metrics:
  - `InstAcc`: valid instruction/tool-call format
  - `ToolAcc`: tool-name selection
  - `ArgAcc`: exact argument-name/value prediction
  - `SummAcc`: final-answer summarization
- End-to-end metrics:
  - `AnsAcc`: whitelist/blacklist answer correctness
  - `P`, `O`, `L`, `C`: tool-selection F1 by category
- Per-task JSON inspection

This is an evaluation utility rather than a heavyweight AgentLego/OpenCompass
execution environment. It is dependency-free and does not require API keys.

## Run

```bash
cd General-Tool-Agent

# Print one official task and its executable ground-truth chain
python3 gta.py --task 0

# Replay the official chain to verify all metric implementations
python3 gta.py --validate

python3 -m unittest -v
```

## Prediction format

Step-by-step predictions map task IDs to lists:

```json
{
  "0": [
    {
      "name": "ImageDescription",
      "arguments": {"image": "image/image_1.jpg"},
      "valid_format": true,
      "summary": "..."
    }
  ]
}
```

End-to-end predictions map task IDs to:

```json
{"0": {"answer": "12", "tools": ["ImageDescription", "OCR", "Calculator"]}}
```

Pass these prediction structures to `evaluate_step_by_step` and
`evaluate_end_to_end` respectively. The upstream repository additionally
provides real AgentLego tool deployment and OpenCompass model execution.

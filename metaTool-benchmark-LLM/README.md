# MetaTool benchmark baseline

This project is a small implementation of the ideas in **MetaTool Benchmark for
Large Language Models: Deciding Whether to Use Tools and Which to Use**
(Huang et al., ICLR 2024; [paper](https://arxiv.org/abs/2310.03128);
[reference implementation](https://github.com/HowieHwong/MetaTool)).

The paper separates two capabilities:

1. **Tool usage awareness**: decide whether the request needs an external tool.
2. **Tool selection**: select the best tool(s) from a candidate set.

The benchmark has four selection settings: similar tools, scenario-specific
tools, reliability-confusable tools, and multi-tool requests. The original
repository generates prompts and evaluates an LLM using a Milvus embedding
store. This starter version keeps the important concepts but is intentionally
dependency-free:

- `LexicalSelector` is an inspectable TF-IDF token-overlap baseline.
- `build_prompt` produces the same numbered tool-list prompt style used by the
  paper's implementation.
- `evaluate` reports awareness accuracy, exact selection accuracy, top-k recall,
  and per-task results.
- `tools.json` is a small catalog that can be replaced by an OpenAI-style tool
  catalog or a name-to-description JSON object.

## Run it

```bash
cd metaTool-benchmark-LLM
python3 metatool.py --query "What is the weather in Hyderabad?"
python3 metatool.py --query "Calculate 12 * 8" --prompt
python3 -m unittest -v
```

Evaluate a JSON or CSV file. JSON records use `query`, plus `tool` for a
single-tool gold label, `tools` for a multi-tool label, and optionally
`use_tool` and `task`:

```bash
python3 metatool.py --dataset examples.json --max-tools 2
```

This is a baseline and not a claim that lexical matching reproduces an LLM.
To reproduce the paper's model experiments, replace `LexicalSelector.decide`
with an LLM adapter and retain the same `evaluate` contract. The original
Milvus/Hugging Face pipeline is deliberately not required for understanding or
testing the benchmark mechanics.

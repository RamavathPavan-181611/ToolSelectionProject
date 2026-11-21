# LLMAgent-Fail-Debug

A local, dependency-free implementation of **“Where LLM Agents Fail and How They
Can Learn From Failures”** ([paper](https://arxiv.org/abs/2509.25370),
[AgentDebug repository](https://github.com/ulab-uiuc/AgentDebug)).

## Implemented concepts

- AgentErrorTaxonomy with Memory, Reflection, Planning, Action, and System
  modules.
- Trajectory parsing for `messages` or `chat_history`.
- Phase 1 fine-grained step/module analysis.
- Phase 2 critical root-cause detection.
- Actionable correction guidance.
- Step accuracy and critical error all-correct evaluation.
- JSON-compatible data structures matching the upstream detector shapes.

The included `HeuristicErrorDetector` is a transparent baseline, not a full
LLM judge. It is suitable for learning, smoke tests, and replacing one method
with a provider model. The upstream repository additionally provides real
provider adapters and full environment rollouts.

## Run

```bash
cd LLMAgent-Fail-Debug

python3 agentdebug.py --trajectory example_trajectory.json
python3 -m unittest -v
```

## Trajectory format

```json
{
  "metadata": {
    "task_id": "demo-1",
    "task": "Put the apple in the microwave",
    "success": false,
    "environment": "alfworld"
  },
  "messages": [
    {"role": "user", "content": "Put the apple in the microwave"},
    {"role": "assistant", "content": "<memory>...</memory><reflection>...</reflection><plan>...</plan><action>...</action>"},
    {"role": "user", "content": "Environment feedback"}
  ]
}
```

Evaluation gold records are the same trajectory records with:

- `step_analyses`: module-level gold results
- `critical_error`: gold critical tuple

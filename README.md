# Tool Selection Project

This repository contains four Python research and prototype projects for
studying tool selection, general tool agents, failure diagnosis, and
self-debugging LLM agents.

## Projects

| Project | Description |
|---|---|
| [Self-Debugging-LLM-ToolAgent](./Self-Debugging-LLM-ToolAgent/) | AgentShield MVP combining MetaTool routing, AgentDebug diagnosis, and GTA-style execution tracing |
| [General-Tool-Agent](./General-Tool-Agent/) | General tool-agent benchmark implementation and GTA reference dataset |
| [LLMAgent-Fail-Debug](./LLMAgent-Fail-Debug/) | Agent failure taxonomy and module-level debugging prototype |
| [metaTool-benchmark-LLM](./metaTool-benchmark-LLM/) | MetaTool benchmark implementation for tool awareness and tool selection |

## Requirements

- Python 3
- Optional: Ollama for the local LLM provider
- Optional: a Gemini API key for the hosted provider fallback

## Running the Projects

Run each project from its own directory:

```bash
cd Self-Debugging-LLM-ToolAgent
python3 agent.py "Calculate 12 * 8"
python3 -m unittest -v
```

```bash
cd General-Tool-Agent
python3 -m unittest -v
```

```bash
cd LLMAgent-Fail-Debug
python3 -m unittest -v
```

```bash
cd metaTool-benchmark-LLM
python3 -m unittest -v
```

See each project directory for project-specific usage and details.

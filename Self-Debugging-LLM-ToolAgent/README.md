# Self-Debugging-LLM-ToolAgent

**AgentShield** is a runnable local project that combines three research ideas:

- **MetaTool**: decide whether external tools are needed and rank/select tools.
- **AgentDebug**: detect module-level failures and identify the earliest critical
  error with corrective guidance.
- **GTA**: expose tool calls, arguments, observations, final answers, and
  end-to-end execution reports.

The implementation is intentionally dependency-free. It is an MVP research
prototype rather than a hosted LLM or real-tool deployment.

## Features

- Confidence-scored hybrid tool routing
- Intent phrases, aliases, regex patterns, argument hints, and description similarity
- Local LLM used for direct answers
- Safe arithmetic calculator
- Workspace-restricted text file reader
- Text statistics and line counting
- Public web search through DuckDuckGo/Wikipedia fallback APIs
- Local image inspection and Tesseract OCR
- SQLite execution history and persistent memory
- Step-by-step execution plan
- Module-level diagnostic errors
- Critical root-cause feedback
- JSON API
- Dependency-free browser interface
- CLI execution
- Free local Ollama provider, with Gemini API fallback option

## Architecture

```text
User Query / Supplied Text
          |
          v
MetaTool Awareness + Ranking
          |
          v
      Tool Plan
          |
          v
Argument Validation + Safe Execution
          |
          v
AgentDebug Module Diagnosis
          |
          v
Final Answer + Tool Trace + Correction
```

## Run the CLI

```bash
cd Self-Debugging-LLM-ToolAgent

python3 agent.py "Calculate 12 * 8"
python3 agent.py "Count words and sentences" --text "Hello world. This is a test."
python3 agent.py "Read example.txt"
python3 -m unittest -v
```

## Free LLM provider

The default provider is **llama.cpp**, running locally without API charges.

```bash
brew services start ollama
ollama pull llama3.2:1b

python3 agent.py "Explain what an API is"
python3 server.py --host 127.0.0.1 --port 8080
```

You can select another Ollama model:

```bash
ollama pull qwen2.5:3b
python3 agent.py "Explain tool calling" --provider ollama --model qwen2.5:3b
python3 server.py --provider ollama --model qwen2.5:3b --port 8080
```

For the optional hosted Gemini free tier:

```bash
export GEMINI_API_KEY="your-free-tier-api-key"
python3 agent.py "Explain tool calling" --provider gemini
python3 server.py --provider gemini --port 8080
```

Check provider availability:

```bash
curl http://127.0.0.1:8080/api/status
```

## Run the Web Interface

```bash
cd Self-Debugging-LLM-ToolAgent
python3 server.py --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080`.

## API

```http
POST /api/run
Content-Type: application/json

{"query":"Calculate 12 * 8","text":""}
```

Response fields include `use_tool`, `selected_tools`, `plan`, `tool_calls`,
`errors`, `critical_error`, `answer`, and `recovered`.

### Persistent history API

```text
GET    /api/history?limit=50
GET    /api/history/:id
GET    /api/memory?limit=100
DELETE /api/history
```

## Supported tools

| Tool | Category | Purpose |
|---|---|---|
| calculator | Logic | Safe arithmetic expressions |
| file_reader | Operation | UTF-8 files inside the workspace |
| text_statistics | Logic | Character, word, line, and sentence counts |
| count_lines | Logic | Count lines in supplied text |
| web_search | Perception | Public web search |
| image_inspector | Perception | Image dimensions, format, and metadata |
| ocr_image | Perception | Local text extraction from uploaded images |

## Multimodal image API

```bash
curl -F "image=@image.png" http://127.0.0.1:8080/api/upload
```

```http
POST /api/run
{"query":"Inspect image uploads/image.png","text":""}
```

```http
POST /api/run
{"query":"OCR image uploads/image.png","text":""}
```

## Project structure

- `agent.py`: unified agent pipeline
- `server.py`: JSON API and static web server
- `static/`: browser interface
- `test_agent.py`: core behavior tests
- `example.txt`: safe file-reader example

## Research references

- MetaTool, arXiv:2310.03128
- AgentDebug, arXiv:2509.25370
- GTA-1, arXiv:2407.08713

## Limitations

This MVP uses deterministic MetaTool routing, heuristic AgentDebug rules, SQLite
persistence, and local/public tools. It does not include multimodal processing,
authentication, or production deployment. Replace provider and tool adapters
before moving to a production implementation.

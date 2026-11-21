#!/usr/bin/env python3
"""Dependency-free local web server for the self-debugging agent."""

from __future__ import annotations

import json
import os
import re
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from agent import SelfDebuggingAgent
from memory_store import MemoryStore


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)


class Handler(BaseHTTPRequestHandler):
    agent = SelfDebuggingAgent(
        ROOT,
        provider=os.getenv("LLM_PROVIDER", "llama_cpp"),
        model=os.getenv("LLM_MODEL"),
    )
    store = MemoryStore(ROOT / "agentshield.db")

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/"}:
            path = "/index.html"
        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content_type = "text/html"
        if target.suffix == ".css":
            content_type = "text/css"
        elif target.suffix == ".js":
            content_type = "application/javascript"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"", "/"}:
            path = "/index.html"
        if path.startswith("/api/"):
            if path == "/api/tools":
                from agent import TOOLS
                self.send_json(200, {
                    "tools": [
                        {
                            "name": tool.name,
                            "category": tool.category,
                            "description": tool.description,
                            "parameters": list(tool.parameters),
                        }
                        for tool in TOOLS
                    ]
                })
                return
            if path == "/api/status":
                status = self.agent.provider_status
                self.send_json(200, {
                    "provider": status.provider,
                    "model": status.model,
                    "available": status.available,
                    "detail": status.detail,
                })
                return
            if path == "/api/history":
                params = parse_qs(urlparse(self.path).query)
                try:
                    limit = int(params.get("limit", ["50"])[0])
                except ValueError:
                    limit = 50
                self.send_json(200, {
                    "executions": self.store.list_executions(limit),
                })
                return
            if path.startswith("/api/history/"):
                try:
                    execution_id = int(path.rsplit("/", 1)[1])
                except ValueError:
                    self.send_json(404, {"error": "execution ID must be numeric"})
                    return
                execution = self.store.get_execution(execution_id)
                if execution is None:
                    self.send_json(404, {"error": "execution not found"})
                    return
                self.send_json(200, execution)
                return
            if path == "/api/memory":
                params = parse_qs(urlparse(self.path).query)
                try:
                    limit = int(params.get("limit", ["100"])[0])
                except ValueError:
                    limit = 100
                self.send_json(200, {"memory": self.store.list_memory(limit)})
                return
            self.send_json(404, {"error": "unknown API endpoint"})
            return

        target = (STATIC / path.lstrip("/")).resolve()
        if STATIC not in target.parents or not target.is_file():
            self.send_error(404)
            return
        content_type = "text/html"
        if target.suffix == ".css":
            content_type = "text/css"
        elif target.suffix == ".js":
            content_type = "application/javascript"
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/upload":
            try:
                content_type = self.headers.get("Content-Type", "")
                if "multipart/form-data" not in content_type:
                    self.send_json(400, {"error": "multipart/form-data is required"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length > 10_000_000:
                    self.send_json(413, {"error": "image exceeds 10 MB"})
                    return
                body = self.rfile.read(length)
                message = BytesParser(
                    policy=policy.default
                ).parsebytes(
                    b"Content-Type: " + content_type.encode("ascii") + b"\r\n"
                    b"MIME-Version: 1.0\r\n\r\n" + body
                )
                saved = []
                for part in message.iter_parts():
                    if part.get_content_disposition() != "form-data":
                        continue
                    field_name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()
                    if field_name != "image" or not filename:
                        continue
                    filename = re.sub(r"[^A-Za-z0-9._-]", "_", Path(filename).name)
                    if not filename or filename in {".", ".."}:
                        continue
                    suffix = Path(filename).suffix.lower()
                    if suffix not in {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff", ".bmp", ".gif", ".webp"}:
                        self.send_json(400, {"error": "unsupported image format"})
                        return
                    payload = part.get_payload(decode=True)
                    if not payload:
                        continue
                    if len(payload) > 10_000_000:
                        self.send_json(413, {"error": "image exceeds 10 MB"})
                        return
                    destination = UPLOADS / filename
                    counter = 1
                    while destination.exists():
                        destination = UPLOADS / f"{Path(filename).stem}_{counter}{suffix}"
                        counter += 1
                    destination.write_bytes(payload)
                    saved.append({
                        "filename": destination.name,
                        "path": f"uploads/{destination.name}",
                        "bytes": len(payload),
                    })
                if not saved:
                    self.send_json(400, {"error": "no image file was provided"})
                    return
                self.send_json(200, {"uploaded": saved})
            except (ValueError, OSError) as exc:
                self.send_json(400, {"error": str(exc)})
            return
        if path != "/api/run":
            self.send_json(404, {"error": "unknown API endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            request = json.loads(self.rfile.read(length).decode("utf-8"))
            query = str(request.get("query", "")).strip()
            supplied_text = str(request.get("text", ""))
            if not query:
                self.send_json(400, {"error": "query is required"})
                return
            report = self.agent.execute(query, supplied_text)
            execution_id = self.store.save_execution(report)
            self.store.save_execution_details(execution_id, supplied_text)
            if report.answer:
                self.store.save_memory(
                    execution_id,
                    "answer",
                    report.answer[:200_000],
                )
            report_payload = dict(report.__dict__)
            report_payload["execution_id"] = execution_id
            report_payload["persisted"] = True
            self.send_json(200, report_payload)
        except json.JSONDecodeError:
            self.send_json(400, {"error": "request body must be valid JSON"})
        except Exception as exc:
            self.send_json(500, {"error": str(exc)})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/history":
            self.send_json(404, {"error": "unknown API endpoint"})
            return
        self.store.clear()
        self.send_json(200, {"cleared": True})

    def log_message(self, format: str, *args: object) -> None:
        print(f"[server] {format % args}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--provider", choices=("llama_cpp", "ollama", "gemini"), default=os.getenv("LLM_PROVIDER", "llama_cpp"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL"))
    args = parser.parse_args()
    Handler.agent = SelfDebuggingAgent(ROOT, provider=args.provider, model=args.model)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"AgentShield running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

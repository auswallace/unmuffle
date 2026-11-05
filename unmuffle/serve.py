"""A local drag-and-drop page: drop a file, watch progress, listen and read.

    unmuffle serve            # opens http://127.0.0.1:8765

Everything stays on this machine: the server listens on 127.0.0.1 only, and the
page loads nothing from the internet. Jobs run one at a time, in a folder
(default ~/unmuffle-jobs), one subfolder per upload.
"""

from __future__ import annotations

import json
import mimetypes
import queue
import re
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import audio, log, pipeline

# rough share of total time per stage, for the progress bar
STAGES = [
    ("read", "Reading the file", 0.02),
    ("tracks", "Cleaning the audio four ways", 0.30),
    ("parts", "Finding clean and noisy parts", 0.05),
    ("speakers", "Finding who speaks when", 0.08),
    ("turns", "Judging and transcribing each turn", 0.35),
    ("check", "Checking every line with a second model", 0.17),
    ("write", "Writing files", 0.03),
]
_NAME_OK = re.compile(r"[^A-Za-z0-9._ -]+")


@dataclass
class Job:
    id: str
    name: str
    folder: Path
    options: dict
    state: str = "queued"          # queued, running, done, error
    stage: str = "read"
    stage_done: float = 0.0        # 0..1 within the current stage
    log: list[str] = field(default_factory=list)
    error: str = ""
    created: float = field(default_factory=time.time)
    finished: float | None = None

    @property
    def source(self) -> Path:
        return self.folder / self.name

    @property
    def out(self) -> Path:
        return self.folder / "result"

    def progress(self) -> float:
        if self.state == "done":
            return 1.0
        total, before = sum(w for *_, w in STAGES), 0.0
        for key, _, w in STAGES:
            if key == self.stage:
                return min(0.99, (before + w * self.stage_done) / total)
            before += w
        return 0.0

    def on_log(self, msg: str) -> None:
        self.log.append(msg)
        m = msg.strip()
        if m.startswith("reading"):
            self.stage, self.stage_done = "read", 0.0
        elif m.startswith(("building", "two-mic")):
            self.stage = "tracks"
            self.stage_done = min(0.9, self.stage_done + 0.3)
        elif "chunks; scoring" in m:
            self.stage, self.stage_done = "parts", 0.0
        elif m.startswith("speaker diarization"):
            self.stage, self.stage_done = "speakers", 0.5 if "noisy" in m else 0.0
        elif "speaker turns to transcribe" in m:
            self.stage, self.stage_done = "turns", 0.0
        elif self.stage == "turns" and re.fullmatch(r"\d+/\d+", m):
            i, n = map(int, m.split("/"))
            self.stage_done = i / max(1, n)
        elif m.startswith("checking"):
            self.stage, self.stage_done = "check", 0.0
        elif m.startswith("check ") and "/" in m:
            i, n = map(int, m.split()[1].split("/"))
            self.stage_done = i / max(1, n)
        elif m.startswith("writing"):
            self.stage, self.stage_done = "write", 0.0

    def summary(self) -> dict:
        return {"id": self.id, "name": self.name, "state": self.state, "stage": self.stage,
                "progress": round(self.progress(), 3), "error": self.error, "created": self.created,
                "finished": self.finished, "options": self.options}

    def detail(self) -> dict:
        d = self.summary()
        d["stages"] = [{"key": k, "label": label} for k, label, _ in STAGES]
        d["log"] = self.log[-40:]
        if self.state == "done":
            d["result"] = self.result()
        return d

    def result(self) -> dict:
        out = self.out
        lines = json.loads((out / "transcript.json").read_text()) if (out / "transcript.json").exists() else []
        report = json.loads((out / "report.json").read_text()) if (out / "report.json").exists() else {}
        speakers = []
        for ln in lines:
            who = ln["speaker"]
            if who not in [s["name"] for s in speakers]:
                speakers.append({"name": who, "file": pipeline.outputs.speaker_filename(who)})
        players = [{"key": "original", "label": "Original", "file": "original.wav"},
                   {"key": "cleaned", "label": "Cleaned", "file": "cleaned.wav"}]
        players += [{"key": f"spk{i}", "label": s["name"], "file": s["file"]}
                    for i, s in enumerate(speakers) if (out / s["file"]).exists()]
        files = sorted(p.name for p in out.iterdir() if p.is_file()) if out.exists() else []
        keep = {"duration_s", "loudness_jump", "clean_seconds", "speakers_found", "methods_used",
                "lines", "low_confidence_lines", "elapsed_s", "dropped_fragments"}
        return {"lines": [{k: ln[k] for k in ("start", "end", "text", "speaker", "speaker_sure", "confidence", "alt")}
                          for ln in lines],
                "players": players, "speakers": [s["name"] for s in speakers], "files": files,
                "report": {k: v for k, v in report.items() if k in keep}}


class Worker:
    """Runs jobs one at a time in a background thread."""

    def __init__(self, root: Path, runner=None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue[Job] = queue.Queue()
        self.runner = runner or self._run_pipeline
        self._load_existing()
        threading.Thread(target=self._loop, daemon=True).start()

    def _load_existing(self) -> None:
        for meta in sorted(self.root.glob("*/job.json")):
            try:
                d = json.loads(meta.read_text())
                job = Job(d["id"], d["name"], meta.parent, d.get("options", {}), state=d.get("state", "done"),
                          created=d.get("created", 0), finished=d.get("finished"), error=d.get("error", ""))
                if job.state in ("queued", "running"):  # interrupted by a restart: run it again (it resumes)
                    job.state = "queued"
                    self.q.put(job)
                self.jobs[job.id] = job
            except (OSError, ValueError, KeyError):
                continue

    def _save(self, job: Job) -> None:
        (job.folder / "job.json").write_text(json.dumps(
            {"id": job.id, "name": job.name, "options": job.options, "state": job.state,
             "created": job.created, "finished": job.finished, "error": job.error}))

    def submit(self, name: str, data_path: Path, options: dict) -> Job:
        jid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        folder = self.root / jid
        folder.mkdir(parents=True)
        safe = _NAME_OK.sub("_", Path(name).name).strip() or "upload"
        data_path.rename(folder / safe)
        job = Job(jid, safe, folder, options)
        self.jobs[jid] = job
        self._save(job)
        self.q.put(job)
        return job

    def _loop(self) -> None:
        while True:
            job = self.q.get()
            job.state = "running"
            self._save(job)
            log.add_listener(job.on_log)
            try:
                self.runner(job)
                job.state = "done"
            except Exception:
                job.state, job.error = "error", traceback.format_exc(limit=3)
            finally:
                log.remove_listener(job.on_log)
                job.finished = time.time()
                self._save(job)

    @staticmethod
    def _run_pipeline(job: Job) -> None:
        o = job.options
        names = tuple(n.strip() for n in o.get("names", "").split(",") if n.strip()) or None
        opts = pipeline.Options(names=names, speakers=o.get("speakers") or None,
                                check_model=None if o.get("fast") else "medium.en")
        pipeline.run(job.source, job.out, opts)
        # a browser-friendly copy of the original, for side-by-side listening
        subprocess.run([audio.require_ffmpeg(), "-y", "-v", "error", "-i", str(job.source), "-vn", "-ac", "1",
                        "-ar", "16000", str(job.out / "original.wav")], check=True)


def _page() -> bytes:
    return resources.files("unmuffle").joinpath("web/index.html").read_bytes()


def make_handler(worker: Worker):
    class Handler(BaseHTTPRequestHandler):
        server_version = "unmuffle"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, obj, code: int = 200) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlparse(self.path)
            parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
            if not parts:
                body = _page()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parts == ["api", "jobs"]:
                self._json([j.summary() for j in sorted(worker.jobs.values(), key=lambda j: -j.created)])
            elif len(parts) == 3 and parts[:2] == ["api", "jobs"] and parts[2] in worker.jobs:
                self._json(worker.jobs[parts[2]].detail())
            elif len(parts) == 5 and parts[:2] == ["api", "jobs"] and parts[3] == "file" and parts[2] in worker.jobs:
                self._file(worker.jobs[parts[2]], parts[4])
            else:
                self._json({"error": "not found"}, 404)

        def _file(self, job: Job, name: str) -> None:
            path = (job.out / name).resolve()
            if path.parent != job.out.resolve() or not path.is_file():
                return self._json({"error": "not found"}, 404)
            size = path.stat().st_size
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            start, end = 0, size - 1
            rng = self.headers.get("Range")
            m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                elif m.group(2):  # last N bytes
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if parse_qs(urlparse(self.path).query).get("download"):
                self.send_header("Content-Disposition", f'attachment; filename="{job.id}-{name}"')
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/api/jobs":
                return self._json({"error": "not found"}, 404)
            name = unquote(self.headers.get("X-Filename", "upload"))
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return self._json({"error": "empty upload"}, 400)
            q = parse_qs(url.query)
            spk = (q.get("speakers") or [""])[0]
            options = {"names": (q.get("names") or [""])[0][:200],
                       "speakers": int(spk) if spk.isdigit() and 0 < int(spk) < 20 else None,
                       "fast": (q.get("fast") or ["0"])[0] == "1"}
            tmp = worker.root / f".upload-{uuid.uuid4().hex}"
            with open(tmp, "wb") as f:
                left = length
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        break
                    f.write(chunk)
                    left -= len(chunk)
            job = worker.submit(name, tmp, options)
            self._json(job.summary(), 201)

    return Handler


def serve(port: int = 8765, root: str | Path | None = None, open_browser: bool = True, runner=None):
    root = Path(root) if root else Path.home() / "unmuffle-jobs"
    worker = Worker(root, runner)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(worker))
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    print(f"unmuffle is running at {url}  (jobs in {root}; Ctrl+C to stop)")
    if open_browser:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    return httpd, worker


def main(port: int, root: str | None, open_browser: bool) -> None:
    httpd, _ = serve(port, root, open_browser)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")

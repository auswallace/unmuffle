import json
import threading
import time
import urllib.request

import numpy as np
import soundfile as sf

from unmuffle import log, serve


def fake_run(job):
    """Stands in for the pipeline: emits the same progress messages and writes the same files."""
    for msg in ["reading x", "building dpdfnet track (dpdfnet)", "6 chunks; scoring the raw audio",
                "speaker diarization (clean part)", "4 speaker turns to transcribe", "  2/4", "  4/4",
                "checking 4 lines with medium.en", "writing cleaned audio"]:
        log.log(msg)
    out = job.out
    out.mkdir(parents=True, exist_ok=True)
    tone = (np.sin(np.arange(16000) / 16000 * 2 * np.pi * 220) * 0.1).astype(np.float32)
    for name in ("cleaned.wav", "original.wav", "speaker_sam.wav"):
        sf.write(out / name, tone, 16000)
    lines = [{"start": 0.0, "end": 1.0, "text": "Hello there.", "track": "raw", "speaker": "Sam",
              "speaker_sure": True, "confidence": 0.9, "alt": "", "meta": {}}]
    (out / "transcript.json").write_text(json.dumps(lines))
    (out / "report.json").write_text(json.dumps({"duration_s": 1.0, "lines": 1, "elapsed_s": 2}))


def _req(url, data=None, headers=None):
    r = urllib.request.Request(url, data=data, headers=headers or {}, method="POST" if data is not None else "GET")
    with urllib.request.urlopen(r) as resp:
        return resp.status, dict(resp.headers), resp.read()


def test_upload_progress_results_and_range(tmp_path):
    httpd, worker = serve.serve(port=0, root=tmp_path, open_browser=False, runner=fake_run)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        status, _, page = _req(base + "/")
        assert status == 200 and b"Drop an audio or video file" in page

        status, _, body = _req(base + "/api/jobs?names=Sam&speakers=1", data=b"\x00" * 1000,
                               headers={"X-Filename": "my%20clip.wav"})
        job = json.loads(body)
        assert status == 201 and job["name"] == "my clip.wav" and job["options"]["speakers"] == 1

        for _ in range(50):
            detail = json.loads(_req(f"{base}/api/jobs/{job['id']}")[2])
            if detail["state"] == "done":
                break
            time.sleep(0.05)
        assert detail["state"] == "done" and detail["progress"] == 1.0
        res = detail["result"]
        assert [p["label"] for p in res["players"]] == ["Original", "Cleaned", "Sam"]
        assert res["lines"][0]["text"] == "Hello there."

        status, headers, chunk = _req(f"{base}/api/jobs/{job['id']}/file/cleaned.wav", headers={"Range": "bytes=0-99"})
        assert status == 206 and len(chunk) == 100 and headers["Accept-Ranges"] == "bytes"
    finally:
        httpd.shutdown()


def test_file_route_blocks_paths_outside_the_job(tmp_path):
    httpd, worker = serve.serve(port=0, root=tmp_path, open_browser=False, runner=fake_run)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        job = json.loads(_req(base + "/api/jobs", data=b"x" * 10, headers={"X-Filename": "a.wav"})[2])
        for _ in range(50):
            if worker.jobs[job["id"]].state == "done":
                break
            time.sleep(0.05)
        try:
            _req(f"{base}/api/jobs/{job['id']}/file/..%2Fjob.json")
            raise AssertionError("expected 404")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        httpd.shutdown()


def test_progress_parsing_moves_forward():
    job = serve.Job("x", "a.wav", None, {})
    seen = []
    msgs = ["reading a.wav", "building gtcrn track (gtcrn)", "12 chunks; scoring", "speaker diarization (clean part)",
            "40 speaker turns to transcribe", "20/40", "40/40", "checking 50 lines", "check 40/50", "writing cleaned"]
    for msg in msgs:
        job.on_log(msg)
        seen.append(job.progress())
    assert seen == sorted(seen) and seen[-1] > 0.9

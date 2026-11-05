"""Command line: unmuffle run | watch | models."""

from __future__ import annotations

import argparse
import os

from . import __version__, models, pipeline, watch


def _opts(a: argparse.Namespace) -> pipeline.Options:
    names = tuple(n.strip() for n in a.names.split(",")) if a.names else None
    methods = tuple(m.strip() for m in a.methods.split(",")) if a.methods else pipeline.METHODS
    bad = [m for m in methods if m not in pipeline.METHODS]
    if bad or "raw" not in methods:
        raise SystemExit(f"--methods must include raw and come from: {', '.join(pipeline.METHODS)}")
    return pipeline.Options(
        model=a.model, check_model=None if a.check == "none" else a.check, threads=a.threads,
        names=names, speakers=a.speakers, methods=methods, shortlist=a.shortlist, write_tracks=a.write_tracks)


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", default="turbo", choices=list(models.WHISPER), help="main Whisper model")
    p.add_argument("--check", default="medium.en", choices=[*models.WHISPER, "none"],
                   help="second model used to flag uncertain lines (default medium.en)")
    p.add_argument("--names", help='speaker names, in the order they first speak: "Alex,Sam"')
    p.add_argument("--speakers", type=int, help="number of speakers (default: detect)")
    p.add_argument("--methods",
                   help=f"cleaning methods to try, comma-separated (default: {','.join(pipeline.METHODS)})")
    p.add_argument("--shortlist", type=int, default=2, help="methods transcribed per noisy chunk (default 2)")
    p.add_argument("--write-tracks", action="store_true", help="also write every method's full track, to compare")
    p.add_argument("--threads", type=int, default=os.cpu_count() or 4)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="unmuffle",
        description="Clean up bad audio and transcribe it: cleaned audio, one file per speaker, "
                    "and a checked transcript.")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="process one file")
    r.add_argument("file")
    r.add_argument("--out", help="output folder (default: <file>.unmuffle next to the input)")
    _common(r)

    w = sub.add_parser("watch", help="process every file dropped into a folder")
    w.add_argument("folder")
    w.add_argument("--interval", type=float, default=5.0)
    w.add_argument("--once", action="store_true", help="process what is there, then exit")
    _common(w)

    s = sub.add_parser("serve", help="open a drag-and-drop page in the browser (local only)")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--jobs", help="folder for runs (default ~/unmuffle-jobs)")
    s.add_argument("--no-browser", action="store_true", help="do not open the browser")

    m = sub.add_parser("models", help="download models ahead of time")
    m.add_argument("--whisper", nargs="*", default=["turbo", "medium.en"])
    m.add_argument("--denoiser", nargs="*", default=["dpdfnet", "gtcrn"])
    m.add_argument("--no-diarization", action="store_true")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        pipeline.run(a.file, a.out, _opts(a))
    elif a.cmd == "watch":
        watch.watch(a.folder, _opts(a), a.interval, a.once)
    elif a.cmd == "serve":
        from . import serve
        serve.main(a.port, a.jobs, not a.no_browser)
    elif a.cmd == "models":
        for name in a.whisper:
            print(f"whisper {name}: {models.whisper_files(name)['encoder'].parent}")
        for name in a.denoiser:
            print(f"denoiser {name}: {models.denoiser_file(name)}")
        if not a.no_diarization:
            print(f"diarization: {models.diarization_files()[0].parent}")
        print(f"dnsmos: {models.dnsmos_file()}")


if __name__ == "__main__":
    main()

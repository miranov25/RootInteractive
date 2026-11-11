"""Narrated review video of a dashboard test run (PHASE_4_7, Claude20.RI). Optional: routine runs never need it.

    python -m RootInteractive.InteractiveDrawing.bokeh.headless.legacy.video RUNDIR [OUT.mp4]

RUNDIR is the test's evidence folder (headless_out/legacy/h01) after a run with RI_DEMO_VIDEO=1. It holds the real
Chromium recording (video/*.webm), timeline.json (when each action started and each checkpoint was read, with its
status), and expected_C*.json / actual_C*.json (the numpy oracle and the page's own values). The video is built from
those only: every number shown or spoken comes from the expected and actual files, every status from the comparator.

For each checkpoint: the recording while the action happens, then the frame at the checkpoint held while the
narration speaks (the test pauses 0.8 s after each checkpoint when it records, so that frame shows the checked state),
with the expected and actual values and the status on screen. The recording is scaled to 1024x576 at the top left, so
the whole dashboard, sliders included, stays visible; the text sits in a panel on the right and a band at the bottom. An introduction and a summary of
passed, known-issue and failed checkpoints frame it. Output: an MP4 (H.264 1280x720, AAC), a JSON script with every
spoken and shown line, and the tools used. Needs ffmpeg and an offline speech engine (espeak-ng, or the
espeakng-loader pip package); without them it stops and says so, it never fakes a video.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import textwrap
import wave
from pathlib import Path

FPS = 25
W, H = 1280, 720                 # output; the browser recording is scaled to PW x PH at the top left, so the whole
PW, PH = 1024, 576               # dashboard stays visible: text goes in the right panel and the bottom band only
PANEL = "0x10202f"
STATUS_COLOR = {"PASS": "0x1b7f3b", "KNOWN_ISSUE_RED": "0xb35c00", "FAIL": "0xb00020", "INFO": "0x2f4f6f"}
STATUS_WORD = {"PASS": "PASS", "KNOWN_ISSUE_RED": "KNOWN ISSUE", "FAIL": "FAIL", "INFO": "REAL TEST RUN"}


# ------------------------------------------------------------------ tools
def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("video: ffmpeg not found (install ffmpeg) - no video produced")


def font_file() -> str:
    if shutil.which("fc-match"):
        f = subprocess.run(["fc-match", "-f", "%{file}", "DejaVu Sans"], capture_output=True, text=True).stdout
        if f and Path(f).exists():
            return f
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
              "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if Path(f).exists():
            return f
    try:
        import matplotlib
        f = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if f.exists():
            return str(f)
    except ImportError:
        pass
    raise SystemExit("video: no DejaVu Sans font found (install dejavu-sans-fonts) - no video produced")


def speak(text: str, wav: Path) -> str:
    """Offline speech into a WAV file; returns the engine used."""
    exe = shutil.which("espeak-ng")
    if exe:
        subprocess.run([exe, "-v", "en-us", "-s", "150", "-w", str(wav), text], check=True, capture_output=True)
        first = (subprocess.run([exe, "--version"], capture_output=True, text=True).stdout.strip().splitlines() or [""])[0]
        return "espeak-ng: " + first[:80]
    try:
        import ctypes
        import espeakng_loader
    except ImportError:
        raise SystemExit("video: no offline speech engine (install espeak-ng) - no video produced")
    lib = ctypes.CDLL(espeakng_loader.get_library_path())
    rate = lib.espeak_Initialize(2, 0, str(Path(espeakng_loader.get_data_path()).parent).encode(), 0)
    chunks = []
    cb_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)
    cb = cb_type(lambda w, n, e: (chunks.append(ctypes.string_at(w, n * 2)) if n > 0 else None) or 0)
    lib.espeak_SetSynthCallback(cb)
    lib.espeak_SetVoiceByName(b"en-us")
    lib.espeak_SetParameter(1, 150, 0)
    raw = text.encode()
    lib.espeak_Synth(raw, len(raw) + 1, 0, 0, 0, 0x1000, None, None)
    lib.espeak_Synchronize()
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(rate), w.writeframes(b"".join(chunks))
    return "espeak-ng library from the pip package espeakng-loader"


def seconds(wav: Path) -> float:
    with wave.open(str(wav)) as w:
        return w.getnframes() / w.getframerate()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ what is said and shown
def _n(x) -> str:
    return f"{int(x):,}"


def narration(cid: str, status: str, exp: dict, act: dict, rows: int) -> str:
    """The spoken sentence for one checkpoint; numbers from the expected and actual files, the verdict from status."""
    e, a = exp["selected_count"], act["selected_count"]
    lo, hi = exp["widgets"]["A"]
    verdict = {"PASS": "Every bin agrees with the reference.",
               "KNOWN_ISSUE_RED": "This is a known issue.",
               "FAIL": "This does not agree with the reference."}[status]
    say = {
        "C1": f"This is the original dashboard. All {_n(rows)} rows are selected; numpy expects {_n(e)}, "
              f"the page shows {_n(a)}. {verdict}",
        "C2": f"We move the A range with the arrow keys, to {lo:.3f} to {hi:.3f}. Numpy expects {_n(e)} selected rows; "
              f"the page shows {_n(a)}. {verdict}",
        "C3": f"We open the Cumulative tab. The cumulative counts end at {_n(act['cumulative'][-1])}, numpy expects "
              f"{_n(exp['cumulative'][-1])}; the distribution ends at {act['cdf'][-1]:.0f}. {verdict}",
        "C4": f"We move both handles back with the arrow keys. Numpy expects all {_n(e)} rows; the page shows {_n(a)}. "
              + ("The upper handle stops a tiny step short of its end, so the row with the largest A stays "
                 "excluded. This is known issue H 0 1 D 1." if status == "KNOWN_ISSUE_RED" else verdict),
        "C5": f"We press End on the upper handle. Numpy expects {_n(e)} rows; the page shows {_n(a)}. {verdict}",
        "C6": f"We go back to the Histograms tab. Every number must equal the original state: numpy expects "
              f"{_n(e)} rows, the page shows {_n(a)}. {verdict}",
    }
    return say.get(cid, f"Checkpoint {cid}: numpy expects {_n(e)} rows, the page shows {_n(a)}. {verdict}")


# ------------------------------------------------------------------ ffmpeg pieces
class Builder:
    def __init__(self, ff: str, font: str, work: Path):
        self.ff, self.font, self.work, self.clips, self.n = ff, font, work, [], 0

    def _text(self, lines) -> Path:
        self.n += 1
        p = self.work / f"text_{self.n:03d}.txt"
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def _draw(self, lines, *, x: str, y: str, size: int, box: str = None, color: str = "white") -> str:
        t = self._text(lines)
        boxed = f":box=1:boxcolor={box}:boxborderw=10" if box else ""
        return (f"drawtext=fontfile='{self.font}':textfile='{t}':fontsize={size}:fontcolor={color}:line_spacing=6"
                f"{boxed}:x={x}:y={y}")

    def _layout(self, title, lines, status: str) -> str:
        """Recording top left; right panel: scene title (wrapped) and status; bottom band: expected and actual."""
        side = []
        for t in title:
            side += textwrap.wrap(t, 21) + [""]
        parts = [f"scale={PW}:{PH},pad={W}:{H}:0:0:color={PANEL}",
                 self._draw(side, x=f"{PW + 16}", y="24", size=20)]
        if status:
            parts.append(self._draw([STATUS_WORD.get(status, status)], x=f"{PW + 16}", y=f"{PH - 60}", size=24,
                                    box=STATUS_COLOR.get(status, "0x333333")))
        if lines:
            parts.append(self._draw(lines, x="20", y=f"{PH + 18}", size=21))
        return ",".join(parts)

    def _run(self, args):
        subprocess.run([self.ff, "-hide_banner", "-loglevel", "error", "-y"] + args, check=True)

    def _out(self) -> Path:
        p = self.work / f"clip_{len(self.clips):03d}.mp4"
        self.clips.append(p)
        return p

    ENC = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-r", str(FPS),
           "-c:a", "aac", "-ar", "44100", "-ac", "2"]

    def action(self, webm: Path, t0: float, t1: float, title):
        t0, t1 = max(0.0, t0), max(t1, t0 + 0.6)
        vf = self._layout(title, ["the action is happening in the real browser..."], None)
        self._run(["-ss", f"{t0:.3f}", "-to", f"{t1:.3f}", "-i", str(webm), "-f", "lavfi", "-i",
                   "anullsrc=r=44100:cl=stereo", "-vf", vf, "-map", "0:v", "-map", "1:a", "-shortest"]
                  + self.ENC + [str(self._out())])

    def hold(self, webm: Path, t: float, title, lines, status: str, wav: Path):
        frame = self.work / f"frame_{len(self.clips):03d}.png"
        self._run(["-i", str(webm), "-ss", f"{max(0.0, t):.3f}", "-frames:v", "1", str(frame)])
        dur = seconds(wav) + 0.8
        self._run(["-loop", "1", "-framerate", str(FPS), "-t", f"{dur:.3f}", "-i", str(frame), "-i", str(wav),
                   "-filter_complex", "[1:a]apad,aresample=44100[a]", "-vf", self._layout(title, lines, status),
                   "-map", "0:v", "-map", "[a]", "-t", f"{dur:.3f}"] + self.ENC + [str(self._out())])
        return dur

    def concat(self, out: Path):
        lst = self.work / "clips.txt"
        lst.write_text("".join(f"file '{c}'\n" for c in self.clips))
        self._run(["-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)])


def build(rundir: Path, out: Path = None) -> dict:
    rundir = Path(rundir)
    tl = json.loads((rundir / "timeline.json").read_text())
    if not tl.get("video"):
        raise SystemExit("video: the run recorded no browser video (run the test with RI_DEMO_VIDEO=1)")
    webm = rundir / "video" / tl["video"]
    out = Path(out) if out else rundir / f"{tl['scenario'].lower()}_review.mp4"
    ff, font = ffmpeg_exe(), font_file()
    events = tl["events"]
    checkpoints = [e for e in events if e["kind"] == "checkpoint"]
    starts = {e["id"]: e["t"] for e in events if e["kind"] == "action"}
    rows = tl["rows"]
    script, engine = [], None
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        b = Builder(ff, font, work)
        intro = (f"This is the review recording of test {tl['scenario']}: RootInteractive's own histogram dashboard, "
                 f"built by its original function on a reproducible table of {_n(rows)} rows, operated in Chromium. "
                 f"Every expected value was computed beforehand with numpy from the raw table.")
        wav = work / "intro.wav"
        engine = speak(intro, wav)
        first = checkpoints[0]["t"] if checkpoints else 0.5
        lines = [f"function: {json.loads((rundir / 'expected_C1.json').read_text())['function']}",
                 f"table: {_n(rows)} rows, seed {tl['seed']}; expected values: numpy, computed before the browser ran",
                 "actual values: the page's own models and drawn plots in Chromium; same comparator as the test"]
        b.hold(webm, first + 0.3, [f"{tl['scenario']}: RootInteractive example dashboard, end to end"], lines,
               "INFO", wav)
        script.append({"scene": "intro", "spoken": intro, "shown": lines})
        prev = first
        for c in checkpoints:
            cid = c["id"]
            exp = json.loads((rundir / f"expected_{cid}.json").read_text())
            act = json.loads((rundir / f"actual_{cid}.json").read_text())
            title = [f"{tl['scenario']} {cid}: {c['scene']}", f"action: {c['action']}"]
            if cid != checkpoints[0]["id"]:
                b.action(webm, starts.get(cid, prev) - 0.2, c["t"] + 0.3, title)
            said = narration(cid, c["status"], exp, act, rows)
            wav = work / f"{cid}.wav"
            speak(said, wav)
            b.hold(webm, c["t"] + 0.4, title, c["overlay"], c["status"], wav)     # inside the 0.8 s demo pause
            script.append({"scene": cid, "status": c["status"], "spoken": said, "shown": c["overlay"],
                           "t_checkpoint": c["t"]})
            prev = c["t"]
        counts = {s: sum(1 for c in checkpoints if c["status"] == s) for s in ("PASS", "KNOWN_ISSUE_RED", "FAIL")}
        health = tl.get("health", {})
        known = sorted(set(health.get("known_defects", {}).values()))
        summary = (f"Summary: {counts['PASS']} of {len(checkpoints)} checkpoints pass"
                   + (f", {counts['KNOWN_ISSUE_RED']} shows a known issue, {', '.join(known)}" if counts['KNOWN_ISSUE_RED'] else "")
                   + (f", {counts['FAIL']} fail" if counts["FAIL"] else ", none fails")
                   + f". The dashboard's health is {'red' if health.get('dashboard') == 'RED' else 'green'}"
                   + (": a valid user action is broken." if health.get("dashboard") == "RED" else "."))
        lines = [f"checkpoints: {counts['PASS']} PASS, {counts['KNOWN_ISSUE_RED']} KNOWN ISSUE"
                 + (f" ({', '.join(known)})" if known else "") + f", {counts['FAIL']} FAIL",
                 f"dashboard health: {health.get('dashboard', '?')}",
                 "evidence: expected_C*.json, actual_C*.json, comparison.json, dashboard_health.json, pytest report"]
        wav = work / "summary.wav"
        speak(summary.replace("H01-D1", "H 0 1 D 1"), wav)
        last = max(e["t"] for e in events) if events else prev
        b.hold(webm, min(last, prev + 0.4), [f"{tl['scenario']}: result"], lines,
               "PASS" if not counts["FAIL"] and not counts["KNOWN_ISSUE_RED"] else
               "KNOWN_ISSUE_RED" if not counts["FAIL"] else "FAIL", wav)
        script.append({"scene": "summary", "spoken": summary, "shown": lines})
        b.concat(out)
    probe = subprocess.run([ff, "-hide_banner", "-i", str(out)], capture_output=True, text=True).stderr
    record = {"mp4": out.name, "mp4_sha256": sha256(out), "webm": webm.name, "webm_sha256": sha256(webm),
              "speech": engine, "ffmpeg": subprocess.run([ff, "-version"], capture_output=True,
                                                         text=True).stdout.splitlines()[0],
              "font": font, "streams": [l.strip() for l in probe.splitlines() if "Stream" in l or "Duration" in l],
              "script": script}
    (out.parent / (out.stem + "_script.json")).write_text(json.dumps(record, indent=1))
    return record


if __name__ == "__main__":
    r = build(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(json.dumps({k: r[k] for k in ("mp4", "mp4_sha256", "speech", "ffmpeg", "streams")}, indent=1))

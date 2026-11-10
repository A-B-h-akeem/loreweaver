"""Scripted demo of the Loreweaver UI, recorded with OBS.

Drives a clean Chrome window (fresh profile, kiosk mode) through a new adventure and a tour of every tab,
while OBS records only that window through its websocket API. The script writes a timeline of the phases
where the model is thinking or streaming, which make_video.py uses to speed those parts up.

  streamlit run app.py   (with LOREWEAVER_DB_PATH pointing at a fresh demo database)
  python demo/record_demo.py --record          # full take, recorded
  python demo/record_demo.py --turns 1         # rehearsal without OBS
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import tempfile
import time
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
URL = "http://localhost:8501"
THEME = "a drowned lighthouse town where the keeper's ghost bargains with smugglers"
TURNS = [   # (kind, value): "dice" = first offered choice with a roll, "choice" = choice index, "say" = free text
    ("dice", None),
    ("say", "I strike a bargain with the keeper's ghost: I will carry her lantern out of the town if she lets "
            "the smugglers' crew go free."),
]
OBS_NAME = "Loreweaver Demo"

CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const add = () => {
    const c = document.createElement('div');
    c.id = 'demo-cursor';
    c.innerHTML = '<svg width="26" height="26" viewBox="0 0 26 26"><path d="M3 2 L3 21 L8 16 L12 24 L15 22.6 '
      + 'L11 15 L18 15 Z" fill="white" stroke="black" stroke-width="1.6" stroke-linejoin="round"/></svg>';
    Object.assign(c.style, {position: 'fixed', left: '760px', top: '430px', zIndex: 2147483647,
      pointerEvents: 'none', filter: 'drop-shadow(0 1px 2px rgba(0,0,0,.6))'});
    document.documentElement.appendChild(c);
    window.__cursor = (x, y) => { c.style.left = x + 'px'; c.style.top = y + 'px'; };
    window.__ripple = () => {
      const r = document.createElement('div');
      Object.assign(r.style, {position: 'fixed', left: (parseFloat(c.style.left) - 14) + 'px',
        top: (parseFloat(c.style.top) - 14) + 'px', width: '28px', height: '28px', borderRadius: '50%',
        border: '3px solid #e09f6b', zIndex: 2147483646, pointerEvents: 'none',
        transition: 'transform .45s ease-out, opacity .45s ease-out'});
      document.documentElement.appendChild(r);
      requestAnimationFrame(() => { r.style.transform = 'scale(2.2)'; r.style.opacity = '0'; });
      setTimeout(() => r.remove(), 500);
    };
  };
  if (document.documentElement) add(); else document.addEventListener('DOMContentLoaded', add);
})();
"""


# ============================================================================================ OBS
class Recorder:
    """OBS through obs-websocket: its own profile and scene collection, so the user's setup is untouched."""

    def __init__(self, out_dir: Path):
        import obsws_python as obs
        cfg = json.loads((Path(os.environ["APPDATA"]) / "obs-studio/plugin_config/obs-websocket/config.json").read_text())
        self.cl = obs.ReqClient(host="localhost", port=cfg["server_port"], password=cfg["server_password"], timeout=15)
        self.out_dir = out_dir
        self.prev_profile = self.cl.get_profile_list().current_profile_name
        self.prev_collection = self.cl.get_scene_collection_list().current_scene_collection_name

    def setup(self, window_title: str):
        cl = self.cl
        if OBS_NAME in cl.get_profile_list().profiles:
            cl.set_current_profile(OBS_NAME)
        else:
            cl.create_profile(OBS_NAME)
        time.sleep(1)
        for cat, name, val in [("Output", "Mode", "Simple"), ("SimpleOutput", "RecFormat2", "mkv"),
                               ("SimpleOutput", "RecEncoder", "x264"), ("SimpleOutput", "RecQuality", "Small"),
                               ("SimpleOutput", "FilePath", str(self.out_dir))]:
            cl.set_profile_parameter(cat, name, val)
        cl.set_video_settings(30, 1, 1920, 1080, 1920, 1080)
        # OBS only reloads output settings on a profile switch: without this, StartRecord silently does nothing
        cl.set_current_profile(self.prev_profile if self.prev_profile != OBS_NAME else "Untitled")
        time.sleep(1)
        cl.set_current_profile(OBS_NAME)
        time.sleep(1)
        if OBS_NAME in cl.get_scene_collection_list().scene_collections:
            cl.set_current_scene_collection(OBS_NAME)
        else:
            cl.create_scene_collection(OBS_NAME)
        time.sleep(2)
        special = cl.get_special_inputs()            # never record the microphone or system sounds
        for attr in ("mic_1", "mic_2", "mic_3", "mic_4", "desktop_1", "desktop_2"):
            name = getattr(special, attr, None)
            if name:
                cl.set_input_mute(name, True)
        scene = cl.get_current_program_scene().current_program_scene_name
        for inp in cl.get_input_list("window_capture").inputs:
            if inp["inputName"] == "Loreweaver window":
                cl.remove_input("Loreweaver window")
                time.sleep(0.5)
        item = cl.create_input(scene, "Loreweaver window", "window_capture",
                               {"window": f"{window_title}:Chrome_WidgetWin_1:chrome.exe", "priority": 0,
                                "method": 2, "cursor": False, "client_area": True}, True).scene_item_id
        cl.set_scene_item_transform(scene, item, {"boundsType": "OBS_BOUNDS_SCALE_INNER", "boundsWidth": 1920.0,
                                                  "boundsHeight": 1080.0, "boundsAlignment": 0,
                                                  "positionX": 0.0, "positionY": 0.0})

    def start(self):
        self.cl.start_record()
        for _ in range(20):
            if self.cl.get_record_status().output_active:
                return
            time.sleep(0.25)
        raise RuntimeError("OBS did not start recording")

    def stop(self) -> str:
        path = self.cl.stop_record().output_path
        time.sleep(2)
        return path

    def restore(self):
        try:
            self.cl.set_current_scene_collection(self.prev_collection)
            time.sleep(1)
            self.cl.set_current_profile(self.prev_profile)
        except Exception as e:  # noqa: BLE001
            print("could not restore OBS profile/collection:", e)


def chrome_window_title(fragment: str) -> str | None:
    """Exact title of the visible Chrome window whose title contains `fragment` (for OBS window matching)."""
    user32, found = ctypes.windll.user32, []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            cls = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(hwnd, cls, 256)
            if fragment in buf.value and cls.value == "Chrome_WidgetWin_1":
                found.append(buf.value)
        return True
    user32.EnumWindows(cb, 0)
    return found[0] if found else None


# ============================================================================================ demo driver
class Demo:
    def __init__(self, page: Page):
        self.page, self.x, self.y = page, 760.0, 430.0
        self.t0: float | None = None
        self.marks: list[dict] = []

    def now(self) -> float:
        return time.time() - self.t0 if self.t0 else 0.0

    @contextmanager
    def phase(self, kind: str):
        start = self.now()
        yield
        self.marks.append({"kind": kind, "start": round(start, 2), "end": round(self.now(), 2)})

    # -------------------------------------------------------------- pointer
    def glide(self, x: float, y: float, dur: float = 0.55):
        steps = max(8, int(dur * 45))
        x0, y0 = self.x, self.y
        for i in range(1, steps + 1):
            f = i / steps
            f = f * f * (3 - 2 * f)
            cx, cy = x0 + (x - x0) * f, y0 + (y - y0) * f
            self.page.mouse.move(cx, cy)
            self.page.evaluate("([x, y]) => window.__cursor && window.__cursor(x, y)", [cx, cy])
            time.sleep(dur / steps)
        self.x, self.y = x, y

    def click(self, loc, pause: float = 0.5, dx: float = 0.5):
        loc.scroll_into_view_if_needed()
        time.sleep(0.25)
        b = loc.bounding_box()
        self.glide(b["x"] + b["width"] * dx, b["y"] + b["height"] / 2)
        time.sleep(0.2)
        self.page.evaluate("window.__ripple && window.__ripple()")
        self.page.mouse.click(self.x, self.y)
        time.sleep(pause)

    def type(self, loc, text: str, delay: int = 38):
        self.click(loc, pause=0.3)
        self.page.keyboard.type(text, delay=delay)
        time.sleep(0.5)

    def scroll(self, total: int, at: tuple[float, float] | None = None, step: int = 60, delay: float = 0.025):
        if at:
            self.glide(*at)
        n = max(1, abs(total) // step)
        for _ in range(n):
            self.page.mouse.wheel(0, step if total > 0 else -step)
            time.sleep(delay)

    def to_bottom(self):
        self.page.mouse.move(self.x, self.y)
        for _ in range(6):
            self.page.mouse.wheel(0, 2500)
            time.sleep(0.05)

    def main_point(self) -> tuple[float, float]:
        vw = self.page.evaluate("window.innerWidth")
        vh = self.page.evaluate("window.innerHeight")
        return vw * 0.62, vh * 0.55

    def tab(self, name: str):
        self.click(self.page.get_by_role("tab", name=name), pause=1.5)

    def pick(self, select_loc, index: int, contains: str | None = None):
        self.click(select_loc, pause=0.8)
        opts = self.page.get_by_role("option")
        if contains and opts.filter(has_text=contains).count():
            opt = opts.filter(has_text=contains).first
        else:
            opt = opts.nth(index)
        self.click(opt, pause=1.2)

    # -------------------------------------------------------------- story
    def main_text_len(self) -> int:
        return self.page.evaluate("(document.querySelector('[data-testid=\"stMain\"]') || document.body).innerText.length")

    def wait_scene(self, turn: int, timeout: float = 420):
        """Model time: 'think' until narration starts streaming, then 'stream' until the choices appear."""
        target = self.page.locator(f".st-key-choice_{turn}_0")
        base, start, t_stream = self.main_text_len(), self.now(), None
        deadline = time.time() + timeout
        last_follow = 0.0
        while time.time() < deadline:
            if target.count() and target.first.is_visible():
                break
            if t_stream is None and self.main_text_len() - base > 350:
                t_stream = self.now()
            if time.time() - last_follow > 1.2:          # follow the streaming text like a reader would
                self.to_bottom()
                last_follow = time.time()
            time.sleep(0.25)
        else:
            raise TimeoutError(f"turn {turn} did not finish")
        end = self.now()
        if t_stream is None:
            t_stream = end
        self.marks.append({"kind": "think", "start": round(start, 2), "end": round(t_stream, 2)})
        self.marks.append({"kind": "stream", "start": round(t_stream, 2), "end": round(end, 2)})
        time.sleep(0.8)
        self.to_bottom()

    def read_scene(self, seconds: float = 4.0):
        """Scroll up through the new scene and back down to the choices, as a reader would."""
        x, y = self.main_point()
        self.glide(x, y, dur=0.4)
        self.scroll(-900, step=45, delay=0.03)
        time.sleep(seconds / 2)
        self.scroll(1200, step=45, delay=0.03)
        self.to_bottom()
        time.sleep(seconds / 2)

    def choose(self, turn: int, kind: str, value):
        # act only once the last scene is committed: the page reruns when the commit finishes, and a rerun
        # mid-click shifts the layout (a click meant for the text box once landed on a choice button)
        self.wait_memory()
        self.to_bottom()
        time.sleep(0.5)
        if kind == "say":
            box = self.page.get_by_placeholder("Or describe anything else you do…")
            self.type(box, value, delay=30)
            self.page.keyboard.press("Enter")
            return
        buttons = self.page.locator(f"[class*='st-key-choice_{turn}_'] button")
        idx = value or 0
        if kind == "dice":
            texts = [buttons.nth(i).inner_text() for i in range(buttons.count())]
            idx = next((i for i, t in enumerate(texts) if " DC " in t), 0)
        self.click(buttons.nth(min(idx, buttons.count() - 1)), pause=0.3)

    def wait_memory(self, timeout: float = 300):
        with self.phase("think"):
            self.page.get_by_text("Memory up to date").first.wait_for(timeout=timeout * 1000)
            time.sleep(1.5)


def run(record: bool, turns: int, hold: float):
    out_dir = HERE / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = Recorder(out_dir) if record else None
    profile = tempfile.mkdtemp(prefix="lw-demo-chrome-")
    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            profile, channel="chrome", headless=False, no_viewport=True,
            ignore_default_args=["--enable-automation"],
            args=["--kiosk", "--no-first-run", "--no-default-browser-check", "--disable-infobars",
                  "--hide-crash-restore-bubble", "--disable-features=Translate"])
        ctx.add_init_script(CURSOR_JS)
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto(URL)
        page.get_by_text("Begin a new tale").wait_for(timeout=60_000)
        time.sleep(3)
        d = Demo(page)
        page.evaluate(CURSOR_JS)

        if rec:
            title = chrome_window_title("Loreweaver")
            if not title:
                raise RuntimeError("Chrome window not found for OBS")
            rec.setup(title)
            time.sleep(2)
            rec.start()
        d.t0 = time.time()
        try:
            time.sleep(hold)                                       # home screen
            d.glide(*d.main_point(), dur=0.8)
            d.type(page.get_by_label("Theme"), THEME, delay=34)
            time.sleep(0.8)
            d.click(page.get_by_role("button", name="Begin adventure"), pause=0.5)
            d.wait_scene(0)                                        # world building + opening scene
            d.read_scene(3)

            for i, (kind, value) in enumerate(TURNS[:turns], start=1):
                d.choose(i - 1, kind, value)
                d.wait_scene(i)
                d.read_scene(3)

            d.wait_memory()                                        # last scene committed to memory
            d.to_bottom()
            sb = page.locator("[data-testid='stSidebar']")         # the sheet and threat clocks
            sbb = sb.bounding_box()
            d.glide(sbb["x"] + sbb["width"] * 0.5, sbb["y"] + 300)
            d.scroll(450, step=45)
            time.sleep(2)
            d.scroll(-450, step=60)

            d.tab("Memory graph")
            frame = page.locator("iframe").first
            frame.wait_for(timeout=30_000)
            time.sleep(3)
            fb = frame.bounding_box()
            cx, cy = fb["x"] + fb["width"] / 2, fb["y"] + fb["height"] / 2
            d.glide(cx, cy, dur=0.7)
            for _ in range(4):                                     # zoom in on the graph
                page.mouse.wheel(0, -120)
                time.sleep(0.2)
            page.mouse.down()                                      # pan across it
            d.glide(cx - 200, cy - 80, dur=1.0)
            page.mouse.up()
            time.sleep(2)

            d.tab("Codex")
            page.get_by_text("Relationships", exact=True).first.wait_for(timeout=30_000)
            time.sleep(1)
            d.pick(page.locator("[data-testid='stSelectbox']").filter(has_text="Entry").first, 1, contains="Character")
            time.sleep(2)
            d.scroll(700, at=d.main_point(), step=50)
            time.sleep(2.5)

            d.scroll(-3000, step=200)
            d.tab("Recall inspector")
            time.sleep(1.5)
            d.type(page.get_by_label("Query"), "What bargain did I make?", delay=35)
            d.click(page.get_by_role("button", name="Recall"), pause=1.0)
            with d.phase("think"):
                running = page.locator("[data-testid='stStatusWidget']")
                deadline = time.time() + 90
                while running.count() and running.first.is_visible() and time.time() < deadline:
                    time.sleep(0.3)
                time.sleep(1)
            d.scroll(600, at=d.main_point(), step=50)
            time.sleep(3)

            d.scroll(-4000, step=200)
            d.tab("Director")
            time.sleep(2.5)
            d.scroll(600, at=d.main_point(), step=50)
            time.sleep(3)

            d.scroll(-4000, step=200)
            d.tab("Story")
            time.sleep(1)
            d.to_bottom()
            time.sleep(2)
        finally:
            path = rec.stop() if rec else None
            (HERE / "timeline.json").write_text(json.dumps({"recording": path, "marks": d.marks}, indent=1))
            if rec:
                rec.restore()
            ctx.close()
    print("recording:", path)
    print("timeline:", HERE / "timeline.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--turns", type=int, default=len(TURNS))
    ap.add_argument("--hold", type=float, default=3.0, help="seconds on the home screen before typing")
    a = ap.parse_args()
    run(a.record, a.turns, a.hold)

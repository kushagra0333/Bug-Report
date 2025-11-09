# bugfiner_runner.py

import os
import re
import sys
import json
import time
import shutil
import platform
import datetime
import subprocess
import asyncio
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

import google.generativeai as genai
from playwright.async_api import async_playwright

# =========================
# ====== CONFIG AREA ======
# =========================

OUTPUT_DIR = "./bug_reports/"
REPORT_DIR = "./bug_reports/"
TEST_NAME_DEFAULT = "LoginButton_Disappears_FailedAttempt"
<<<<<<< HEAD
LOCAL_HTML_DEFAULT = "http://localhost:8000/login.html"   # change at runtime with --url
Gemini_api_key="AIzaSyA5hnjXwU0QfBkc7JMJbQ-izK1dwX_qG8E"
GEMINI_API_KEY = Gemini_api_key          # overridable via --gemini-key or env
=======
LOCAL_HTML_DEFAULT = "http://localhost:8000/login.html"
Gemini_api_key="Enter_API_Key_here"
GEMINI_API_KEY = Gemini_api_key
>>>>>>> 08b10d8b3fed61ccb5889993ba40c78d63cf7c8c
VIDEO_SIZE = {"width": 1280, "height": 720}
HEADLESS_DEFAULT = True
RECORD_VIDEO_DEFAULT = True
SCREENSHOT_EVERY_STEP = True  # set True if you want a screenshot at each step

# Make sure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# =========================
# ====== UTILITIES ========
# =========================

def now_stamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_text_utf8(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def clean_json(text: str) -> str:
    """Remove markdown fences like ```json ... ``` or stray backticks."""
    txt = text.strip()
    txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    return txt.strip()

def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def build_env_info() -> Dict[str, Any]:
    try:
        import playwright
        pw_version = getattr(playwright, "__version__", "unknown")
    except Exception:
        pw_version = "unknown"
    return {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "playwright": pw_version,
        "node_hint": "Playwright bundles Chromium; Node not required for Python runner",
        "ffmpeg_available": has_ffmpeg(),
        "time": datetime.datetime.now().isoformat(timespec="seconds")
    }

def levenshtein(a: str, b: str) -> int:
    """Simple Levenshtein distance for fuzzy selector matching."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur_row = [i]
        for j, cb in enumerate(b, 1):
            insert = prev_row[j] + 1
            delete = cur_row[j - 1] + 1
            subst = prev_row[j - 1] + (0 if ca == cb else 1)
            cur_row.append(min(insert, delete, subst))
        prev_row = cur_row
    return prev_row[-1]

def score_str(hay: str, needle: str) -> float:
    """Heuristic score for how well hay matches needle."""
    hay = (hay or "").lower()
    needle = (needle or "").lower()
    if not hay and not needle:
        return 0.0
    if needle in hay:
        return 0.1  # very good (substring)
    # normalized Levenshtein (lower is better)
    dist = levenshtein(hay, needle)
    denom = max(len(hay), len(needle), 1)
    return 1.0 * dist / denom

# =========================
# ====== REPORTING ========
# =========================

@dataclass
class RunAttachment:
    kind: str
    path: str

@dataclass
class RunResult:
    test_name: str
    timestamp: str
    steps_human: List[str]
    step_errors: List[str]
    console_logs: List[str]
    environment: Dict[str, Any]
    attachments: List[RunAttachment]

class BugReporter:
    def __init__(self, report_dir: str):
        self.report_dir = report_dir

    def save_report(self, result: RunResult) -> Dict[str, str]:
        ts = result.timestamp
        base = os.path.join(self.report_dir, f"{result.test_name}_{ts}")
        json_path = f"{base}.json"
        md_path = f"{base}.md"

        serializable = {
            **asdict(result),
            "attachments": [asdict(a) for a in result.attachments]
        }
        write_text_utf8(json_path, json.dumps(serializable, indent=2, ensure_ascii=False))

        lines = []
        lines.append(f"# Bug Report: {result.test_name}")
        lines.append(f"**Date:** {result.timestamp}")
        lines.append("")
        lines.append("## Steps Executed")
        for s in result.steps_human:
            lines.append(f"- {s}")
        lines.append("")
        lines.append("## Step Errors")
        if result.step_errors:
            for e in result.step_errors:
                lines.append(f"- ❌ {e}")
        else:
            lines.append("- ✅ No step errors")
        lines.append("")
        lines.append("## Console Logs")
        if result.console_logs:
            for c in result.console_logs:
                lines.append(f"- {c}")
        else:
            lines.append("- (no console logs)")
        lines.append("")
        lines.append("## Environment")
        for k, v in result.environment.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
        lines.append("## Attachments")
        if result.attachments:
            for a in result.attachments:
                lines.append(f"- **{a.kind}**: `{a.path}`")
        else:
            lines.append("- (no attachments)")
        lines.append("")

        write_text_utf8(md_path, "\n".join(lines))
        print(f"📄 Bug report saved:\n- {json_path}\n- {md_path}")
        return {"json": json_path, "md": md_path}

# =========================
# === LLM STEP BUILDER ====
# =========================

ACTION_MAP = {
    "input": "fill",
    "enter text": "fill",
    "type": "fill",
    "assert element not present": "checkElementVisibility",
    "checkElementPresence": "checkElementVisibility",
    "wait": "waitForLoadState",
}

def coerce_steps_to_schema(steps: List[Dict[str, Any]], default_url: str) -> List[Dict[str, Any]]:
    """Normalize actions + force CSS selectors + ensure URL for 'open'."""
    normalized = []
    for s in steps:
        action = s.get("action", "").strip()
        action_norm = ACTION_MAP.get(action, action)
        target = s.get("target", "")
        value = s.get("value", "")

        step = {"action": action_norm}

        if action_norm == "open":
            if not target or target.upper() == "URL" or (not target.startswith("http") and not target.startswith("file://")):
                target = default_url
            step["target"] = target

        elif action_norm in ("click", "fill", "checkElementVisibility"):
            # If target isn't clearly CSS-like, prefix with '#'
            if target and not target.startswith(("#", ".", "xpath=", "text=", "css=")):
                target = "#" + target.lstrip("#")
            step["target"] = target
            if action_norm == "fill":
                step["value"] = value

        elif action_norm == "screenshot":
            step["target"] = target if target else f"screenshot_{int(time.time())}.png"

        elif action_norm == "waitForLoadState":
            pass

        else:
            step = {"action": "_noop", "note": f"Unknown action '{action}' target '{target}'"}

        normalized.append(step)

    # Ensure first action is open
    if not normalized or normalized[0].get("action") != "open":
        normalized.insert(0, {"action": "open", "target": default_url})

    return normalized

def get_llm_steps(bug_description: str, gemini_key: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """Best-effort Gemini prompt; return None on failure or missing key."""
    if not gemini_key:
        return None
    try:
        genai.configure(api_key=gemini_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a QA engineer. Convert the following bug report into structured steps. "
            "Output ONLY valid JSON array (no markdown, no comments) in this format:\n"
            "[{\"action\": \"open\", \"target\": \"URL\"}, "
            "{\"action\": \"fill\", \"target\": \"CSS_SELECTOR\", \"value\": \"text\"}, "
            "{\"action\": \"click\", \"target\": \"CSS_SELECTOR\"}, "
            "{\"action\": \"screenshot\", \"target\": \"filename.png\"}, "
            "{\"action\": \"assertText\", \"target\": \"CSS_SELECTOR\", \"value\": \"expected substring\"}]\n\n"
            f"Bug Report: {bug_description}"
        )
        resp = model.generate_content(prompt)
        raw = clean_json(resp.text or "").strip()
        if not raw:
            return None
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        return None
    except Exception as e:
        print(f"Gemini error (ignored, using fallback): {e}")
        return None

def fallback_steps(default_url: str) -> List[Dict[str, Any]]:
    """Deterministic baseline steps for your local login.html (works with your buggy page)."""
    return [
        {"action": "open", "target": default_url},
        {"action": "fill", "target": "#username", "value": "invalid_username"},  # will be healed
        {"action": "fill", "target": "#password", "value": "invalid_password"},  # will be healed
        {"action": "click", "target": "#login_button"},
        {"action": "assertText", "target": "#message", "value": "Login"},
        {"action": "screenshot", "target": "after_click.png"},
    ]

# =========================
# === SMART SELECTORS  ====
# =========================

class SelfHealingMap:
    """Persist self-healed mappings during a run."""
    def __init__(self):
        self._map: Dict[str, str] = {}

    def set(self, req: str, resolved: str):
        self._map[req] = resolved

    def get(self, req: str) -> Optional[str]:
        return self._map.get(req)

async def best_input_selector(page, hint: str) -> Optional[str]:
    """
    Try to find the best input for a given hint (e.g., '#username').
    Checks id/name/placeholder/aria-label and nearby label text.
    """
    # Direct try first
    if hint.startswith(("css=", "xpath=", "#", ".", "text=", "[", "input", "textarea")):
        try:
            el = await page.query_selector(hint)
            if el:
                return hint
        except:
            pass

    # Candidate scanning
    candidates = await page.query_selector_all("input, textarea, [contenteditable]")
    scored: List[Tuple[float, str]] = []

    for el in candidates:
        attrs = await el.evaluate("""(e) => ({
            id: e.id || "",
            name: e.name || "",
            placeholder: e.placeholder || "",
            aria: e.getAttribute("aria-label") || "",
            type: e.type || ""
        })""")
        id_ = attrs["id"]
        nm = attrs["name"]
        ph = attrs["placeholder"]
        ar = attrs["aria"]
        ty = attrs["type"]

        pool = [id_, nm, ph, ar, ty]
        # Weighted scoring: user/password special-casing
        targets = []
        if "user" in hint.lower():
            targets = ["user", "username", "email", "login"]
        elif "pass" in hint.lower():
            targets = ["pass", "password", "pwd"]

        base_score = min(score_str(" ".join(pool), hint), 1.0)
        if targets:
            boosts = [score_str(" ".join(pool), t) for t in targets]
            base_score = min(base_score, min(boosts))

        # Lower is better
        scored.append((base_score, f"#{id_}" if id_ else None, el, id_, nm, ph, ar, ty))

    # sort by lowest score
    scored.sort(key=lambda x: x[0])
    for sc, sel, el, id_, nm, ph, ar, ty in scored:
        # pick the best we can form a selector for:
        if id_:
            return f"#{id_}"
        if nm:
            return f"[name='{nm}']"
        if ph:
            # Escape quotes in placeholder if any
            safe_ph = ph.replace("'", "\\'")
            return f"[placeholder='{safe_ph}']"
        # As last resort, return the nth-of-type selector
        # But that requires index; we avoid for now.
    return None

async def best_button_selector(page, hint: str) -> Optional[str]:
    """
    Find a button-like element matching 'login', 'submit', CSS id, etc.
    """
    # direct attempt
    try:
        el = await page.query_selector(hint)
        if el:
            return hint
    except:
        pass

    candidates = await page.query_selector_all("button, input[type=button], input[type=submit], [role=button]")
    scored: List[Tuple[float, str]] = []
    for el in candidates:
        attrs = await el.evaluate("""(e) => ({
            id: e.id || "",
            name: e.name || "",
            value: e.value || "",
            text: e.innerText || e.textContent || ""
        })""")
        id_ = attrs["id"]
        nm = attrs["name"]
        val = attrs["value"]
        text = attrs["text"]

        pool = " ".join([id_, nm, val, text])
        # weight login-ish forms
        targets = ["login", "submit", "sign in", "log in"]
        base = score_str(pool, hint)
        boosts = [score_str(pool, t) for t in targets]
        score = min([base]+boosts)

        scored.append((score, id_, nm, val, text))

    scored.sort(key=lambda x: x[0])
    for sc, id_, nm, val, text in scored:
        if id_:
            return f"#{id_}"
        if nm:
            return f"[name='{nm}']"
        if text.strip():
            # Use :has-text() Playwright pseudo for text
            # but query_selector does not support ':has-text' — use locator internally in click.
            # We return text= for convenience
            return f"text={text.strip()}"
    return None

async def smart_fill(page, requested_selector: str, value: str, healer: SelfHealingMap) -> Tuple[bool, str]:
    # respect healed mapping
    healed = healer.get(requested_selector)
    try_first = healed or requested_selector
    try:
        await page.fill(try_first, value)
        if healed:
            return True, healed
        return True, try_first
    except:
        pass

    resolved = await best_input_selector(page, requested_selector)
    if resolved:
        try:
            await page.fill(resolved, value)
            healer.set(requested_selector, resolved)
            return True, resolved
        except Exception as e:
            return False, f"Fill failed after heal: {e}"
    return False, "Fill failed: no matching input found"

async def smart_click(page, requested_selector: str, healer: SelfHealingMap) -> Tuple[bool, str]:
    healed = healer.get(requested_selector)
    try_first = healed or requested_selector
    # Try click via locator if text= syntax
    try:
        if try_first.startswith("text="):
            await page.get_by_text(try_first.replace("text=", ""), exact=True).click()
        else:
            await page.click(try_first)
        if healed:
            return True, healed
        return True, try_first
    except:
        pass

    resolved = await best_button_selector(page, requested_selector)
    if resolved:
        try:
            if resolved.startswith("text="):
                await page.get_by_text(resolved.replace("text=", ""), exact=True).click()
            else:
                await page.click(resolved)
            healer.set(requested_selector, resolved)
            return True, resolved
        except Exception as e:
            return False, f"Click failed after heal: {e}"
    return False, "Click failed: no matching button found"

# =========================
# ====== PLAYWRIGHT =======
# =========================

async def run_steps_playwright(
    steps: List[Dict[str, Any]],
    test_name: str,
    headless: bool,
    record_video: bool,
    timeout_ms: int
) -> RunResult:

    ts = now_stamp()
    log_path = os.path.join(OUTPUT_DIR, f"log_{ts}.txt")
    console_logs: List[str] = []
    step_errors: List[str] = []
    steps_human: List[str] = []
    attachments: List[RunAttachment] = []
    actual_video_path = None
    healer = SelfHealingMap()

    def human(i, s):
        return f"Step {i+1}: {s}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)

        context_kwargs = {
            "record_video_dir": OUTPUT_DIR,
            "viewport": {"width": VIDEO_SIZE["width"], "height": VIDEO_SIZE["height"]},
        }
        if record_video:
            context_kwargs["record_video_size"] = {"width": VIDEO_SIZE["width"], "height": VIDEO_SIZE["height"]}

        context = await browser.new_context(**context_kwargs)
        page = await context.new_page()

        # Capture console logs
        def on_console(msg):
            kind = msg.type
            text = msg.text
            console_logs.append(f"[{kind}] {text}")

        page.on("console", on_console)

        # Step execution
        for i, step in enumerate(steps):
            action = step.get("action")
            target = step.get("target")
            value = step.get("value", "")

            try:
                if action == "_noop":
                    steps_human.append(human(i, f"noop: {step.get('note','')}"))
                    continue

                if action == "open":
                    steps_human.append(human(i, f"open → {target}"))
                    await page.goto(target, wait_until="domcontentloaded", timeout=timeout_ms)

                elif action == "click":
                    steps_human.append(human(i, f"click → {target}"))
                    ok, info = await smart_click(page, target, healer)
                    if not ok:
                        raise RuntimeError(info)

                elif action == "fill":
                    steps_human.append(human(i, f"fill '{value}' → {target}"))
                    ok, info = await smart_fill(page, target, value, healer)
                    if not ok:
                        raise RuntimeError(info)

                elif action == "screenshot":
                    out_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(os.path.basename(target))[0]}_{ts}.png")
                    steps_human.append(human(i, f"screenshot → {out_path}"))
                    await page.screenshot(path=out_path, full_page=True)
                    attachments.append(RunAttachment(kind="screenshot", path=out_path))

                elif action == "waitForLoadState":
                    steps_human.append(human(i, "waitForLoadState(networkidle)"))
                    await page.wait_for_load_state("networkidle")

                elif action == "checkElementVisibility":
                    steps_human.append(human(i, f"assert visibility → {target}"))
                    el = await page.query_selector(target)
                    visible = await el.is_visible() if el else False
                    expected = step.get("expected", True)
                    if visible != expected:
                        step_errors.append(f"Assertion failed: {target} visible={visible}, expected={expected}")

                elif action == "assertText":
                    steps_human.append(human(i, f"assertText '{value}' in {target}"))
                    el = await page.query_selector(target)
                    txt = await el.inner_text() if el else ""
                    if value not in txt:
                        step_errors.append(f"assertText failed: expected '{value}' in '{txt}' at {target}")

                else:
                    steps_human.append(human(i, f"unknown action '{action}' → skip"))

                if SCREENSHOT_EVERY_STEP and action not in ("screenshot", "open"):
                    snap_path = os.path.join(OUTPUT_DIR, f"step_{i+1}_{ts}.png")
                    await page.screenshot(path=snap_path, full_page=True)
                    attachments.append(RunAttachment(kind="screenshot", path=snap_path))

            except Exception as e:
                step_errors.append(f"Step {i+1} failed ({action} {target}): {e}")

        # Write run log
        lines = [f"Bug Run - {ts}", ""]
        if step_errors:
            lines.append("--- Step Errors ---")
            lines += step_errors
            lines.append("")
        if console_logs:
            lines.append("--- Console Logs ---")
            lines += console_logs
            lines.append("")
        if not step_errors and not console_logs:
            lines.append("No console or step errors found.")
        write_text_utf8(log_path, "\n".join(lines))
        attachments.append(RunAttachment(kind="log", path=log_path))

        # video
        await context.close()
        if page.video:
            try:
                actual_video_path = await page.video.path()
            except Exception:
                actual_video_path = None
        await browser.close()

    # Optional ffmpeg conversion
    if actual_video_path and os.path.exists(actual_video_path):
        if has_ffmpeg():
            mp4_path = os.path.join(OUTPUT_DIR, f"video_{ts}.mp4")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", actual_video_path, "-c:v", "libx264", "-c:a", "aac", mp4_path],
                    check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
                attachments.append(RunAttachment(kind="video", path=mp4_path))
                try:
                    os.remove(actual_video_path)
                except Exception:
                    pass
            except Exception:
                attachments.append(RunAttachment(kind="video", path=actual_video_path))
        else:
            attachments.append(RunAttachment(kind="video", path=actual_video_path))

    env = build_env_info()
    result = RunResult(
        test_name=test_name,
        timestamp=ts,
        steps_human=steps_human,
        step_errors=step_errors,
        console_logs=console_logs,
        environment=env,
        attachments=attachments
    )
    return result

# =========================
# ========= MAIN ==========
# =========================

def parse_args():
    ap = argparse.ArgumentParser(description="BugFiner Runner")
    ap.add_argument("--url", default=LOCAL_HTML_DEFAULT, help="Target page URL (http://... or file://...)")
    ap.add_argument("--test-name", default=TEST_NAME_DEFAULT, help="Test name for report")
    ap.add_argument("--headful", action="store_true", help="Run browser with UI")
    ap.add_argument("--no-video", action="store_true", help="Disable video recording")
    ap.add_argument("--steps-json", default="", help="Path to steps JSON file (override LLM/fallback)")
    ap.add_argument("--bug", default="Login button disappears after failed login attempt on Chrome", help="Bug description for LLM")
    ap.add_argument("--gemini-key", default="", help="Gemini API key (overrides code)")
    ap.add_argument("--timeout", type=int, default=30000, help="Per-action timeout in ms")
    return ap.parse_args()

async def main():
    args = parse_args()

    default_url = args.url
    test_name = args.test_name
    headless = not args.headful
    record_video = not args.no_video
    timeout_ms = args.timeout

    gemini_key = args.gemini_key or os.environ.get("GEMINI_API_KEY") or (GEMINI_API_KEY if GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_OR_EMPTY" else "")

    # Step source preference: explicit steps json > LLM > fallback
    if args.steps_json and os.path.exists(args.steps_json):
        with open(args.steps_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
        steps = coerce_steps_to_schema(raw, default_url)
        print("✅ Using steps from file:")
        print(json.dumps(steps, indent=2))
    else:
        print("→ Generating steps from LLM (if API available)...")
        llm_steps = get_llm_steps(args.bug, gemini_key)
        if llm_steps is not None:
            print("Generated Steps JSON (raw):")
            print(json.dumps(llm_steps, indent=2))
            steps = coerce_steps_to_schema(llm_steps, default_url)
        else:
            print("LLM unavailable/failed, using fallback deterministic steps.")
            steps = fallback_steps(default_url)

        print("✅ Final Steps for execution:")
        print(json.dumps(steps, indent=2))

    result = await run_steps_playwright(
        steps=steps,
        test_name=test_name,
        headless=headless,
        record_video=record_video,
        timeout_ms=timeout_ms
    )
    reporter = BugReporter(REPORT_DIR)
    paths = reporter.save_report(result)

    # Summary
    print("\n===== RUN SUMMARY =====")
    print(f"Test: {test_name}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Steps executed: {len(result.steps_human)}")
    print(f"Step errors: {len(result.step_errors)}")
    print(f"Console entries: {len(result.console_logs)}")
    print(f"Report JSON: {paths['json']}")
    print(f"Report MD:   {paths['md']}")
    for a in result.attachments:
        print(f"- {a.kind}: {a.path}")

<<<<<<< HEAD
if __name__ == "__main__":
    asyncio.run(main())
=======

>>>>>>> 08b10d8b3fed61ccb5889993ba40c78d63cf7c8c

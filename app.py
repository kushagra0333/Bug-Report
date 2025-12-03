from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import asyncio
import os
import sys

# Import your bugfiner runner
from bugfiner_runner import main as bugfiner_main, REPORT_DIR

app = Flask(__name__)

# Directories
os.makedirs(REPORT_DIR, exist_ok=True)

MEDIA_DIR = os.path.join(REPORT_DIR)
os.makedirs(MEDIA_DIR, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        bug_desc = request.form.get("bug_desc", "").strip()
        test_url = request.form.get("test_url", "").strip()

        if bug_desc and test_url:
            # Prepare args for bugfiner
            args = [
                "--url", test_url,
                "--test-name", "auto_test",
                "--bug", bug_desc,
            ]

            # Run bugfiner synchronously
            asyncio.run(run_bugfiner(args))

            return redirect(url_for("index"))

    # Load latest report
    latest_report = None
    reports = [
        f for f in sorted(os.listdir(REPORT_DIR), reverse=True) if f.endswith(".md")
    ]
    if reports:
        latest_file = os.path.join(REPORT_DIR, reports[0])
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                latest_report = f.read()
        except Exception as e:
            latest_report = f"⚠️ Error reading report: {e}"

    # Collect gallery items
    gallery_items = []
    if os.path.exists(MEDIA_DIR):
        for f in sorted(os.listdir(MEDIA_DIR)):
            file_extension = os.path.splitext(f)[1].lower()
            if file_extension in (".png", ".jpg", ".jpeg"):
                gallery_items.append({"filename": f, "type": "image"})
            elif file_extension in (".mp4", ".webm"):
                gallery_items.append({"filename": f, "type": "video"})

    return render_template("index.html", report=latest_report, gallery=gallery_items)


@app.route("/media/<path:filename>")
def media(filename):
    """
    Serve media files (screenshots/videos).
    """
    return send_from_directory(MEDIA_DIR, filename)


async def run_bugfiner(args_list):
    """
    Runs bugfiner's main function with sys.argv patched.
    """
    sys.argv = ["bugfiner_runner.py"] + args_list
    await bugfiner_main()


if __name__ == "__main__":
    app.run(debug=True)
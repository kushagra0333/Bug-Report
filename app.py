from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import asyncio
import os
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sys

# Import your bugfiner runner
from bugfiner_runner import main as bugfiner_main, REPORT_DIR

# Load environment variables
load_dotenv()

app = Flask(__name__)

# MongoDB setup
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = None
db = None
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    # Test connection
    client.admin.command("ping")
    db = client["bugfiner_db"]
except ConnectionFailure as e:
    print(f"⚠️ MongoDB connection failed: {e}. Falling back to file-based reporting.")

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
            try:
                asyncio.run(run_bugfiner(args))
            except Exception as e:
                return render_template("index.html", report=f"⚠️ Error running BugFiner: {e}", gallery=[])

            return redirect(url_for("index"))

    # Load latest report
    latest_report = None
    gallery_items = []
    if db:
        # Query MongoDB for the latest report
        reports_collection = db["bug_reports"]
        latest_doc = reports_collection.find().sort("timestamp", -1).limit(1)
        for doc in latest_doc:
            latest_report = doc.get("markdown_report", "⚠️ No report content available")
            # Extract attachments for gallery
            for attachment in doc.get("attachments", []):
                if attachment["kind"] in ("screenshot", "video"):
                    if os.path.exists(os.path.join(MEDIA_DIR, attachment["path"])):
                        gallery_items.append({"filename": attachment["path"], "type": attachment["kind"]})
    else:
        # Fallback to filesystem
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

        # Collect gallery items from filesystem
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
    app.run(debug=os.environ.get("FLASK_ENV") == "development")
import sys
from pathlib import Path
from typing import Dict

# set up directories'
print("Setting up directories...")
from project_config import PROJECT_ROOT

print(f"Project root: {PROJECT_ROOT}")
# models module directory
DIR_MODELS = PROJECT_ROOT / "src"
print(f"Models at {DIR_MODELS}")
sys.path.append(str(DIR_MODELS))
# example data directory
DATA_DIR = PROJECT_ROOT / "data"
print(f"Example data at: {DATA_DIR}")
DATA_PRIVACY_POLICY_DIR = DATA_DIR / "privacy_policy_alexa"

# read a list of available privacy policies
EXAMPLE_PRIVACY_POLICIES: Dict[str, Path] = {
    f"{pp.stem.split('.')[0]}": pp for pp in DATA_PRIVACY_POLICY_DIR.glob("*.md")
}
print(f"Loaded privacy policies: {EXAMPLE_PRIVACY_POLICIES.keys()}")

from flask import Flask, jsonify, make_response, request
from flask_cors import CORS

from src.model import get_extractive_summarizer

app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})

# load up the model
summarizer = get_extractive_summarizer(model_type="distilbert", device="gpu")


@app.route("/api/privacy_policies")
def get_available_privacy_policies():
    return jsonify(list(EXAMPLE_PRIVACY_POLICIES.keys()))


@app.route("/api/summarize", methods=["GET"])
def get_text_summary():
    args = request.args
    if "docType" not in args and "docName" not in args:
        missing_parameters = []
        for p in ["docType", "docName"]:
            if p not in args:
                missing_parameters.append(p)
        return make_response(
            f"Missing query parameters: {', '.join(missing_parameters)}", 400
        )
    doc_type = args["docType"]
    doc_name = args["docName"]
    if doc_type == "pp":
        if doc_name in EXAMPLE_PRIVACY_POLICIES:
            app.logger.info(
                f"Loading privacy policy text {str(EXAMPLE_PRIVACY_POLICIES[doc_name])}"
            )
            document = EXAMPLE_PRIVACY_POLICIES[doc_name].read_text(encoding="utf-8")
            return document
        else:
            return make_response(
                f"The specified document type Privacy Policy does not have file {doc_name} available.",
                400,
            )


if __name__ == "__main__":
    print(PROJECT_ROOT)
    app.run()

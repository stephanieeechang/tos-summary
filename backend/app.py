import sys
from pathlib import Path
from typing import Dict, List

# set up directories'
import torch.cuda

from utils import chunkify_text

print("Setting up directories...")
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
print(f"Project root: {PROJECT_ROOT}")
# models module directory
DIR_MODELS = PROJECT_ROOT / "src"
print(f"Models at {DIR_MODELS}")
sys.path.append(str(DIR_MODELS.absolute()))
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

from src.helpers import summarize_text
from src.model import get_extractive_summarizer

app = Flask(__name__)
CORS(app, resources={"/api/*": {"origins": "*"}})

# load up the model
device_name = "cpu"
if torch.cuda.is_available():
    device_name = "cuda"
torch_device = torch.device(device_name)

print("Loading extractive summarizer...")
summarizer = get_extractive_summarizer(model_type="distilbert", device=device_name)
print("Extractive summarizer loaded.")

@app.route("/api/privacy_policies")
def get_available_privacy_policies():
    return jsonify(list(EXAMPLE_PRIVACY_POLICIES.keys()))


@app.route("/api/summarize", methods=["GET"])
def get_text_summary():
    def stream_summarization(text_chunks: List[str]):
        num_chunks = len(text_chunks)
        for text_index, text in enumerate(text_chunks):
            app.logger.info(f"Summarizing chunk {text_index + 1} of {num_chunks}")
            yield summarize_text(
                text, model=summarizer, device=torch_device, do_print=False
            )
        app.logger.info(f"Streamed {num_chunks} chunks of text.")

    args = request.args
    app.logger.info(f"/api/summarize received arguments: {args}")
    if "custom" in args:  # preferss custom text, although this should not happen
        pass
    else:
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
        if doc_type == "privacy policy":
            if doc_name in EXAMPLE_PRIVACY_POLICIES:
                app.logger.info(
                    f"Loading privacy policy text {str(EXAMPLE_PRIVACY_POLICIES[doc_name])}"
                )
                document = EXAMPLE_PRIVACY_POLICIES[doc_name].read_text(
                    encoding="utf-8"
                )
                app.logger.info(f"Splitting text into chunks of size 500 words...")
                chunkified_document = chunkify_text(document, 500)
                app.logger.info(f"Streaming predictions...")
                return app.response_class(
                    stream_summarization(chunkified_document), mimetype="text/plain"
                )
            else:
                return make_response(
                    f"The specified document type Privacy Policy does not have file {doc_name} available.",
                    400,
                )
        else:
            return make_response(
                f"The specified document type {doc_type} is not supported.", 400
            )


if __name__ == "__main__":
    print(PROJECT_ROOT)
    app.run()

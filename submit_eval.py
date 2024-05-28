import argparse
import json
import os
import glob
from dataclasses import dataclass, make_dataclass
from enum import Enum
from datetime import datetime, timezone
from huggingface_hub import CommitOperationAdd
from leaderboard.envs import API, EVAL_REQUESTS_PATH, TOKEN, QUEUE_REPO, RESULTS_REPO
from leaderboard.check_validity import (
    already_submitted_models,
    check_model_card,
    get_model_size,
    is_model_on_hub,
)

REQUESTED_MODELS = None
USERS_TO_SUBMISSION_DATES = None


@dataclass
class ModelDetails:
    name: str
    display_name: str = ""
    symbol: str = ""  # emoji


class ModelType(Enum):
    PT = ModelDetails(name="pretrained", symbol="🟢")
    FT = ModelDetails(name="fine-tuned", symbol="🔶")
    IFT = ModelDetails(name="instruction-tuned", symbol="⭕")
    RL = ModelDetails(name="RL-tuned", symbol="🟦")
    Unknown = ModelDetails(name="", symbol="?")

    def to_str(self, separator=" "):
        return f"{self.value.symbol}{separator}{self.value.name}"

    @staticmethod
    def from_str(type):
        if "fine-tuned" in type or "🔶" in type:
            return ModelType.FT
        if "pretrained" in type or "🟢" in type:
            return ModelType.PT
        if "RL-tuned" in type or "🟦" in type:
            return ModelType.RL
        if "instruction-tuned" in type or "⭕" in type:
            return ModelType.IFT
        return ModelType.Unknown


def create_request_file(
    model_name, base_model, revision, precision, model_type, weight_type
):
    global REQUESTED_MODELS
    global USERS_TO_SUBMISSION_DATES
    if not REQUESTED_MODELS:
        REQUESTED_MODELS, USERS_TO_SUBMISSION_DATES = already_submitted_models(
            EVAL_REQUESTS_PATH
        )

    model_type = ModelType.from_str(model_type).to_str()

    user_name = ""
    model_path = model_name
    if "/" in model_name:
        user_name = model_name.split("/")[0]
        model_path = model_name.split("/")[1]

    precision = precision.split(" ")[0]
    current_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if model_type is None or model_type == "":
        raise Exception("Please select a model type.")

    # Does the model actually exist?
    if revision == "":
        revision = "main"

    # Is the model on the hub?
    if weight_type in ["Delta", "Adapter"]:
        base_model_on_hub, error, _ = is_model_on_hub(
            model_name=base_model,
            revision=revision,
            token=TOKEN,
            test_tokenizer=True,
            trust_remote_code=True,
        )
        if not base_model_on_hub:
            raise Exception(f'Base model "{base_model}" {error}')

    if not weight_type == "Adapter":
        print("----------------- ", model_name, revision)
        model_on_hub, error, _ = is_model_on_hub(
            model_name=model_name,
            revision=revision,
            token=TOKEN,
            test_tokenizer=True,
            trust_remote_code=True,
        )
        if not model_on_hub:
            raise Exception(f'Model "{model_name}" {error}')

    # Is the model info correctly filled?
    try:
        model_info = API.model_info(repo_id=model_name, revision=revision)
    except Exception:
        raise Exception(
            "Could not get your model information. Please fill it up properly."
        )

    model_size = get_model_size(model_info=model_info, precision=precision)

    # Were the model card and license filled?
    try:
        license = model_info.cardData["license"]
    except Exception:
        raise Exception("Please select a license for your model")

    modelcard_OK, error_msg = check_model_card(model_name)
    if not modelcard_OK:
        raise Exception(error_msg)

    # Seems good, creating the eval
    print("Adding new eval")

    eval_entry = {
        "model": model_name,
        "base_model": base_model,
        "revision": revision,
        "precision": precision,
        "weight_type": weight_type,
        "status": "RUNNING",
        "submitted_time": current_time,
        "model_type": model_type,
        "likes": model_info.likes,
        "params": model_size,
        "license": license,
        "private": False,
    }

    # Check for duplicate submission
    if f"{model_name}_{revision}_{precision}" in REQUESTED_MODELS:
        raise Exception("This model has been already submitted.")

    print("Creating eval file")
    OUT_DIR = f"{EVAL_REQUESTS_PATH}/{user_name}"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = (
        f"{OUT_DIR}/{model_path}_eval_request_False_{precision}_{weight_type}.json"
    )

    with open(out_path, "w") as f:
        f.write(json.dumps(eval_entry))

    print("Uploading eval file")
    API.upload_file(
        path_or_fileobj=out_path,
        path_in_repo=out_path.split("eval-queue/")[1],
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"Add {model_name} to eval queue",
    )

    print("Your request has been submitted to the evaluation queue!")
    return out_path


def evaluate_model(model, model_name, batch_size, revision, output_path, request_file):
    run = f"lm_eval --model={model} --model_args='pretrained={model_name},revision={revision}' --tasks openllm --device cuda:0 --batch_size {batch_size} --output_path={output_path} --trust_remote_code"

    exit_code = os.system(run)
    if exit_code == 0:
        print("Command executed successfully.")
        submit_results(
            model_name=args.model_name,
            output_path=args.output_path,
            request_file=request_file,
        )
    else:
        print(f"Command execution failed. Exit code: {exit_code}")


def submit_results(model_name, output_path):
    def find_json_files(path):
        json_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".json"):
                    json_files.append(os.path.join(root, file))
        return json_files

    def rename_keys(d):
        new_dict = {}
        for key, value in d.items():
            if isinstance(value, dict):
                # Recursively rename keys in nested dictionaries
                value = rename_keys(value)
            # Split the key on ',' and take the first part
            new_key = key.split(",")[0]
            new_dict[new_key] = value
        return new_dict

    # Especificar o caminho do diretório que deseja procurar
    directory_path = f"{output_path}/*{model_name.split('/')[-1]}"
    directory_path = glob.glob(directory_path)

    # Chamar a função e obter a lista de arquivos .json
    json_files_list = []
    for path in directory_path:
        json_files_list.extend(find_json_files(path))

    # Exibir a lista de arquivos .json encontrados
    for json_file in json_files_list:
        with open(json_file) as f:
            d = json.load(f)

        results = {
            "results": rename_keys(d["results"]),
            "versions": d["versions"],
            "config": {
                "model": d["config"]["model"],
                "model_args": d["config"]["model_args"],
                "num_fewshot": 0,
                "batch_size": d["config"]["batch_size"],
                "batch_sizes": d["config"]["batch_sizes"],
                "device": d["config"]["device"],
                "no_cache": True,
                "limit": d["config"]["limit"],
                "bootstrap_iters": d["config"]["bootstrap_iters"],
                "description_dict": None,
                "model_dtype": d["config"]["model_dtype"],
                "model_name": d["model_name"],
                "model_sha": d["config"]["model_sha"],
            },
        }

        with open(json_file, "w") as f:
            json.dump(results, f)

        print("Uploading eval file")
        API.upload_file(
            path_or_fileobj=json_file,
            path_in_repo=json_file.split(output_path)[1],
            repo_id=RESULTS_REPO,
            repo_type="dataset",
            commit_message=f"Add {model_name} to results",
        )

    with open(request_file) as f:
        request = json.load(f)

    request["status"] = "FINISHED"

    with open(request_file, "w") as f:
        f.write(json.dumps(request))

    API.upload_file(
        path_or_fileobj=request_file,
        path_in_repo=request_file.split("eval-queue/")[1],
        repo_id=QUEUE_REPO,
        repo_type="dataset",
        commit_message=f"FINISHED",
    )

    # Remove the local file
    os.remove(request_file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Selects which model type or provider is evaluated. See https://github.com/EleutherAI/lm-evaluation-harness/tree/main#model-apis-and-inference-servers for full list",
        default="huggingface",
        nargs="?",
    )
    parser.add_argument("--model_name", type=str, help="Model Name")
    parser.add_argument(
        "--revision",
        type=str,
        help="Revision Commit",
        default="main",
        nargs="?",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="Precision",
        choices=["float16", "bfloat16", "float32", "8bit", "4bit", "GPTQ"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        help="Model Type",
        choices=["pretrained", "fine-tuned", "instruction-tuned", "RL-tuned"],
    )
    parser.add_argument(
        "--weight_type",
        type=str,
        help="Weights Type",
        default="Original",
        choices=["Original", "Adapter", "Delta"],
        nargs="?",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        help="Base model for delta or adapter weights",
        default="",
        nargs="?",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Output path for save results",
        default="./results",
        nargs="?",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        help="Batch size for model evaluation. Batch size selection can be automated by setting the flag to auto",
        default="auto:4",
        nargs="?",
    )

    args = parser.parse_args()

    request_file = create_request_file(
        model_name=args.model_name,
        revision=args.revision,
        precision=args.precision,
        model_type=args.model_type,
        weight_type=args.weight_type,
        base_model=args.base_model,
    )

    evaluate_model(
        model=args.model,
        model_name=args.model_name,
        batch_size=args.batch_size,
        revision=args.revision,
        output_path=args.output_path,
        request_file=request_file,
    )

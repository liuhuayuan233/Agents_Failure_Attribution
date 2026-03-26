import os
import re
import argparse
import contextlib
import sys
import datetime
import yaml
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI

try:
    import torch
    from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
    HAS_LOCAL_DEPS = True
except ImportError:
    HAS_LOCAL_DEPS = False
    torch = None
    pipeline = None
    AutoModelForCausalLM = None
    AutoTokenizer = None

from Lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search
)

try:
    from Lib.local_model import (
        analyze_all_at_once_local,
        analyze_step_by_step_local,
        analyze_binary_search_local
    )
except ImportError:
    analyze_all_at_once_local = None
    analyze_step_by_step_local = None
    analyze_binary_search_local = None


KNOWN_GPT_MODELS = {"gpt-4o", "gpt4", "gpt4o-mini", "gpt-5.4-nano"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-72b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent chat history using specific models.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default = "../Who_and_When/Algorithm-Generated",
        help="Path to the directory containing JSON chat history files. Default: '../Who_and_When/Algorithm-Generated'."
    )

    parser.add_argument(
        "--is_handcrafted",
        type=str,
        default="False",
        choices=['True', 'False'], # If you want to test Hand-Crafted, set is_handcrafted to be True.
        help="Specify 'True' or 'False'. Default: 'False'."
    )


    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "feedback", "feedback", "prompts", "openai.yaml"),
        help="Path to openai.yaml config file with api_key and api_base."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for API response. Used only for GPT models."
    )

    parser.add_argument(
        "--device", type=str, default="cuda:1" if (torch and torch.cuda.is_available()) else "cpu",
        help="Device for local model inference (e.g., 'cuda', 'cuda:0', 'cpu'). Default: 'cuda' if available, else 'cpu'."
    )

    parser.add_argument(
        "--no_ground_truth",
        action="store_true",
        default=False,
        help="Do not include ground truth answer in the prompt (closed-book mode)."
    )

    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Test mode: only process the first JSON file, print the prompt without calling LLM."
    )

    args = parser.parse_args()

    client_or_model_obj = None
    model_type = None # gpt, llama, qwen
    model_family = None 
    model_id_or_deployment = args.model

    if args.test:
        if args.model in KNOWN_GPT_MODELS:
            model_type = 'gpt'
            model_family = 'gpt'
        elif args.model in LOCAL_MODEL_ALIASES:
            model_type = 'local'
            model_id_or_deployment = LOCAL_MODEL_MAP[args.model]
            model_family = 'llama' if args.model in LOCAL_LLAMA_ALIASES else 'qwen'
        print(f"[TEST MODE] Skipping model initialization for {args.model}")
    elif args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
        model_family = 'gpt'
        print(f"Selected GPT model: {args.model}")

        config_path = os.path.abspath(args.config)
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        with open(config_path, 'r', encoding='utf-8') as f:
            api_config = yaml.safe_load(f)
        api_key = api_config.get("api_key", "")
        api_base = api_config.get("api_base", "")
        if not api_key or not api_base:
            print("Error: api_key and api_base must be set in openai.yaml")
            sys.exit(1)
        try:
            client_or_model_obj = OpenAI(
                api_key=api_key,
                base_url=api_base,
            )
            print(f"Successfully initialized OpenAI client for endpoint: {api_base}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            sys.exit(1)

    elif args.model in LOCAL_MODEL_ALIASES:
        if not HAS_LOCAL_DEPS:
            print("Error: torch and transformers are required for local models. Install them first.")
            sys.exit(1)
        model_type = 'local'
        model_id_or_deployment = LOCAL_MODEL_MAP[args.model]

        if args.model in LOCAL_LLAMA_ALIASES:
            model_family = 'llama'
            print(f"Selected local Llama model: {args.model} ({model_id_or_deployment}) on device {args.device}")
            if not pipeline:
                 print("Error: transformers library not found or pipeline could not be imported.")
                 sys.exit(1)
            try:
                 print(f"Initializing Llama pipeline for {model_id_or_deployment}...")
                 client_or_model_obj = pipeline(
                     "text-generation",
                     model=model_id_or_deployment,
                     model_kwargs={"torch_dtype": torch.bfloat16},
                     device=args.device,
                 )
                 print(f"Successfully initialized Llama pipeline on {args.device}.")
            except Exception as e:
                print(f"Error initializing Llama pipeline for {model_id_or_deployment}: {e}")
                sys.exit(1)

        elif args.model in LOCAL_QWEN_ALIASES:
            model_family = 'qwen'
            print(f"Selected local Qwen model: {args.model} ({model_id_or_deployment}) on device {args.device}")
            if not AutoModelForCausalLM or not AutoTokenizer:
                 print("Error: transformers library not found or specific classes could not be imported.")
                 sys.exit(1)
            try:
                 print(f"Initializing Qwen model and tokenizer for {model_id_or_deployment}...")
                 qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_deployment,
                    torch_dtype="auto",
                    device_map=args.device # Use device_map for potentially large models
                 )
                 qwen_tokenizer = AutoTokenizer.from_pretrained(model_id_or_deployment)
                 client_or_model_obj = (qwen_model, qwen_tokenizer) # Store as tuple
                 print(f"Successfully initialized Qwen model and tokenizer on {args.device}.")
            except Exception as e:
                print(f"Error initializing Qwen model/tokenizer for {model_id_or_deployment}: {e}")
                print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
                sys.exit(1)
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)


    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    handcrafted_suffix = "_handcrafted" if args.is_handcrafted == "True" else "_alg_generated"
    output_filename = f"{args.method}_{args.model.replace('/','_')}{handcrafted_suffix}.txt"
    output_filepath = os.path.join(output_dir, output_filename)
    
    args.is_handcrafted = True if args.is_handcrafted == "True" else False # Update: Convert string to boolean

    # 断点续传：检查已有输出中已完成的文件
    skip_files = set()
    if not args.test and os.path.exists(output_filepath):
        try:
            with open(output_filepath, 'r', encoding='utf-8') as f:
                existing = f.read()
            for m in re.finditer(r"Prediction for ([^:]+\.json):", existing):
                skip_files.add(m.group(1).strip())
            if skip_files:
                print(f"[断点续传] 已有 {len(skip_files)} 条结果，跳过已完成的文件")
        except Exception:
            pass

    print(f"Analysis method: {args.method}")
    print(f"Model Alias: {args.model} (Family: {model_family})")
    if not args.test:
        print(f"Output will be saved to: {output_filepath}")

    def _run_analysis():
        print(f"--- Starting Analysis: {args.method} ---")
        print(f"Timestamp: {datetime.datetime.now()}")
        print(f"Model Family: {model_family}")
        print(f"Model Used: {model_id_or_deployment}")
        print(f"Input Directory: {args.directory_path}")
        print(f"Is Handcrafted: {args.is_handcrafted}")
        print(f"No Ground Truth: {args.no_ground_truth}")
        print(f"Test Mode: {args.test}")
        print("-" * 20)

        if model_type == 'gpt':
            if args.method == "all_at_once":
                gpt_all_at_once(
                    client=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test,
                    skip_files=skip_files
                )
            elif args.method == "step_by_step":
                gpt_step_by_step(
                    client=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test,
                    skip_files=skip_files
                )
            elif args.method == "binary_search":
                gpt_binary_search(
                    client=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test,
                    skip_files=skip_files
                )
        elif model_type == 'local':
            if args.method == "all_at_once":
                analyze_all_at_once_local(
                    model_obj=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model_family=model_family,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test
                )
            elif args.method == "step_by_step":
                analyze_step_by_step_local(
                    model_obj=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model_family=model_family,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test
                )
            elif args.method == "binary_search":
                analyze_binary_search_local(
                    model_obj=client_or_model_obj,
                    directory_path=args.directory_path,
                    is_handcrafted=args.is_handcrafted,
                    model_family=model_family,
                    no_ground_truth=args.no_ground_truth,
                    test_mode=args.test
                )

        else:
             print(f"Internal Error: Unknown model_type '{model_type}' during function call.")

        print("-" * 20)
        print(f"--- Analysis Complete ---")

    try:
        if args.test:
            _run_analysis()
        else:
            write_mode = 'a' if skip_files else 'w'
            with open(output_filepath, write_mode, encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
                _run_analysis()
            print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()
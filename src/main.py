import logging
import os
import random
from argparse import ArgumentParser

import nltk
import numpy as np
import torch

from helpers import summarize, summarize_text, test_rouge
from model import (ALTERNATE_CHECKPOINT_NAME, BERT_BASE_CHECKPOINT_NAME,
                   CHECKPOINT_DIR, ExtSummarizer)

nltk.download("punkt")

logger = logging.getLogger(__name__)

if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir()

for d in [BERT_BASE_CHECKPOINT_NAME, ALTERNATE_CHECKPOINT_NAME]:
    if not d.parent.exists():
        d.parent.mkdir()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.warning(
        "Deterministic mode can have a performance impact, depending on your model. This means "
        + "that due to the deterministic nature of the model, the processing speed (i.e. "
        + "processed batch items per second) can be lower than when the model is "
        + "non-deterministic."
    )


def main(args):
    if args.seed:
        set_seed(args.seed)
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA GPU not available, using CPU.")
            device = torch.device("cpu")
        print("Using device:", device)
    else:
        device = torch.device("cpu")
        print("Using device:", device)
    if args.model_type == "bertbase":
        # if not os.path.exists(
        #         "1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
        # ):
        logger.info(f"Loading checkpoint from {str(BERT_BASE_CHECKPOINT_NAME)}")
        if not BERT_BASE_CHECKPOINT_NAME.exists():
            os.system(
                f'curl "https://www.googleapis.com/drive/v3/files/1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE" -o {str(BERT_BASE_CHECKPOINT_NAME)}'
            )
        checkpoint = torch.load(
            # f"1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE",
            str(BERT_BASE_CHECKPOINT_NAME),
            map_location=device,
        )
    else:
        # if not os.path.exists(
        #         "1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
        # ):
        logger.info(f"Loading checkpoint from {str(ALTERNATE_CHECKPOINT_NAME)}")
        if not ALTERNATE_CHECKPOINT_NAME.exists():
            os.system(
                f'curl "https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE" -o {str(ALTERNATE_CHECKPOINT_NAME)}'
            )
        checkpoint = torch.load(
            # f"1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE",
            str(ALTERNATE_CHECKPOINT_NAME),
            map_location=device,
        )
    logger.info(f"Instantiating summarizer with arguments: {args.model_type}, {device}")
    model = ExtSummarizer(
        checkpoint=checkpoint, bert_type=args.model_type, device=device
    )

    if args.demo_mode:
        text = input("Enter any Terms of Services excerpt: ")
        summarize_text(text, model, device, max_length=2)
    else:
        if not args.result_dir:
            result_fp = f"results/summary_{args.model_type}.json"
            dir = result_fp.split("/")[0] + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            summarize(result_fp, model, device, training=args.do_train, max_length=1)
            print("Summary saved in " + result_fp)
            curr_dict, avg_rouge_legalsum, avg_rouge_tosdr = test_rouge(result_fp)
        else:
            dir = args.result_dir.split("/")[0] + "/"
            if not os.path.exists(dir):
                os.makedirs(dir)
            summarize(
                args.result_dir, model, device, training=args.do_train, max_length=1
            )
            print("Summary saved in " + args.result_dir)
            curr_dict, avg_rouge_legalsum, avg_rouge_tosdr = test_rouge(args.result_dir)

    # TODO: Add ROUGE score calculation functions from helpers.py


if __name__ == "__main__":
    parser = ArgumentParser(add_help=False)

    # parametrize the network: general options
    parser.add_argument(
        "--model_type",
        type=str,
        default="distilbert",
        choices=["distilbert", "bertbase"],
        help="Use Distil BERT or BERT base model. Default is 'distilbert'.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        help="Default path for saving summarization results.",
    )
    parser.add_argument(
        "--gpu",
        default=False,
        type=bool,
        help="Whether GPU is used. Default is False, meaning CPU is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for reproducible results. Can negatively impact performance in some cases.",
    )
    parser.add_argument(
        "--do_train",
        default=False,
        type=bool,
        help="If True, train. If False, test.",
    )
    parser.add_argument(
        "-l",
        "--log",
        dest="logLevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: 'Info').",
    )
    parser.add_argument(
        "--demo_mode",
        action="store_true",
        help="Enable demo mode. Input a text instead of a dataset, and prints summary.",
    )

    main_args = parser.parse_known_args()

    if not main_args[0].result_dir:
        logger.warning(
            "Argument `--save_results` not specified to use save results. Default path 'results/summary_{args.model_type}.json' used."
        )

    main_args = parser.parse_args()

    # Setup logging config
    logging.basicConfig(
        format="%(asctime)s|%(name)s|%(levelname)s> %(message)s",
        level=logging.getLevelName(main_args.logLevel),
    )

    # Run
    main(main_args)

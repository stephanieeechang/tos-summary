import gzip
import json
import logging
import os
import textwrap

import numpy as np
import torch
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def split_dataset(training=True, dataset="legal_summarization"):
    if dataset == "legal_summarization":
        data_dir = os.path.dirname(__file__)
        rel_path = "../data/legal_summarization/all_v1.json"
        abs_data_path = os.path.join(data_dir, rel_path)
        documents, file_path = load_json(abs_data_path)
        doc_list = list(documents)
        # total = 446
        # (train, test, val) = (356, 45, 45)
        #
        if training:
            train_idx, test_idx = train_test_split(
                doc_list, test_size=0.2, random_state=6471
            )
            return train_idx, test_idx, documents
        else:
            _, test_idx = train_test_split(doc_list, test_size=0.9, random_state=6471)
            return test_idx, documents
        # val, test = train_test_split(test, test_size=0.5, random_state=6471)
        # return train, test, val
    elif dataset == "privacy_policy_alexa":
        pass
    else:
        logger.error(
            #     "Dataset name %s not recognized. Please use either 'legal_summarization' or 'privacy_policy_alexa'.",
            "Dataset name %s not recognized. Please use 'legal_summarization'.",
            dataset,
        )


def load_json(json_file):
    """Load a json file even if it is compressed with gzip.

    Args:
        json_file (str): Path to json file

    Returns:
        tuple: (documents, file_path), loaded json and path to file
    """
    # `file_extension` is second and path (without extension) is first
    # `file_extension` only contains last extension so ".json.gz" will output ".gz"
    file_path, file_extension = os.path.splitext(json_file)
    if file_extension == ".json":
        with open(json_file, "r") as json_file_object:
            documents = json.load(json_file_object)
    elif file_extension == ".gz":
        file_path = os.path.splitext(file_path)[0]  # remove ".gz"
        # https://stackoverflow.com/a/39451012
        with gzip.open(json_file, "r") as json_gzip:
            json_bytes = json_gzip.read()
        json_str = json_bytes.decode("utf-8")
        documents = json.loads(json_str)  # "loads": the "s" means string
    else:
        logger.error(
            "File extension %s not recognized. Please use either '.json' or '.gz'.",
            file_extension,
        )
        documents = None
    return documents, file_path


def preprocess(training):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    if training:
        train_idx, test_idx, documents = split_dataset(training)
    else:
        test_idx, documents = split_dataset(training)

    # Get JSON dictionary keys
    doc_list = documents.keys()
    for i in doc_list:
        documents[i]["original_text"].replace("\n", " ").replace("[CLS] [SEP]", " ")
        sents = sent_tokenize(documents[i]["original_text"])
        documents[i]["original_text"] = "[CLS] [SEP]".join(sents)
    return documents


def preprocess_text(text):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    raw_text = text.replace("\n", " ").replace("[CLS] [SEP]", " ")
    sents = sent_tokenize(raw_text)
    processed_text = "[CLS] [SEP]".join(sents)
    return processed_text


def load_text(processed_text, max_pos, device):
    # CONTRACTS-BERT-BASE, trained on US contracts
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/bert-base-uncased-contracts")
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    sep_vid = tokenizer.vocab["[SEP]"]
    cls_vid = tokenizer.vocab["[CLS]"]

    def _process_src(raw):
        raw = raw.strip().lower()
        raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        src_subtokens = tokenizer.tokenize(raw)
        src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = tokenizer.convert_tokens_to_ids(src_subtokens)
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]

        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(processed_text)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    src_text = [
        [sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]
    ]
    return src, mask_src, segs, clss, mask_cls, src_text


def test(model, input_dict, input_data, result_path, max_length, block_trigram=True):
    def _get_ngrams(n, text):
        """Calculates n-grams.

        Args:
            n (int): which n-grams to calculate
            text (list): An array of tokens

        Returns:
            A set of n-grams
        """
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _block_trigrams(candidate, prediction):
        """Decrease repetition in summaries by checking if a trigram from ``prediction``
        exists in ``candidate``

        Args:
            candidate (str): The string to check for trigrams from ``prediction``
            prediction (list): A list of strings to extract trigrams from

        Returns:
            bool: True if overlapping trigrams detected, False otherwise.
        """
        tri_c = _get_ngrams(3, candidate.split())
        for s in prediction:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False

    # If not in demo mode, save results to the path specified
    if input_dict and result_path:
        try:
            with open(result_path, "r") as save_pred:
                curr_dict = json.load(save_pred)
        except (json.decoder.JSONDecodeError, FileNotFoundError) as e:
            curr_dict = {}

        with open(result_path, "w+") as save_pred:
            with torch.no_grad():
                src, mask, segs, clss, mask_cls, src_str = input_data
                sent_scores, mask = model(src, segs, clss, mask, mask_cls)
                sent_scores = sent_scores + mask.float()
                sent_scores = sent_scores.cpu().data.numpy()
                selected_ids = np.argsort(-sent_scores, 1)

                pred = []
                for i, idx in enumerate(selected_ids):
                    _pred = []
                    if len(src_str[i]) == 0:
                        continue
                    for j in selected_ids[i][: len(src_str[i])]:
                        if j >= len(src_str[i]):
                            continue
                        candidate = src_str[i][j].strip()
                        if block_trigram:
                            if not _block_trigrams(candidate, _pred):
                                _pred.append(candidate)
                        else:
                            _pred.append(candidate)

                        if len(_pred) == max_length:
                            break

                    _pred = " ".join(_pred)
                    pred.append(_pred)

                for i in range(len(pred)):
                    pred_str = pred[i].strip() + "\n"

                input_dict["extractive_summary"] = pred_str
                key = str(input_dict["uid"])
                curr_dict[key] = input_dict
                json.dump(curr_dict, save_pred)
    # If in demo mode, print out results
    else:
        with torch.no_grad():
            src, mask, segs, clss, mask_cls, src_str = input_data
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            selected_ids = np.argsort(-sent_scores, 1)

            pred = []
            for i, idx in enumerate(selected_ids):
                _pred = []
                if len(src_str[i]) == 0:
                    continue
                for j in selected_ids[i][: len(src_str[i])]:
                    if j >= len(src_str[i]):
                        continue
                    candidate = src_str[i][j].strip()
                    if block_trigram:
                        if not _block_trigrams(candidate, _pred):
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if len(_pred) == max_length:
                        break

                _pred = " ".join(_pred)
                pred.append(_pred)

            for i in range(len(pred)):
                pred_str = pred[i].strip() + "\n"

            wrapper = textwrap.TextWrapper(width=80)
            print("Summary:")
            print(wrapper.fill(pred_str))


def summarize(result_save_path, model, device, training, max_length=3, max_pos=512):
    model.eval()
    processed_dict = preprocess(training)
    dict_keys = processed_dict.keys()
    for key in tqdm(dict_keys):
        input_dict = processed_dict[key]
        input_data = load_text(input_dict["original_text"], max_pos, device=device)
        test(
            model,
            input_dict,
            input_data,
            result_save_path,
            max_length,
            block_trigram=True,
        )


def summarize_text(text, model, device, max_length=3, max_pos=512):
    model.eval()
    processed_text = preprocess_text(text)
    input_data = load_text(processed_text, max_pos, device=device)
    test(model, None, input_data, None, max_length, block_trigram=True)


def cal_rouge(evaluated_ngrams, reference_ngrams):
    reference_count = len(reference_ngrams)
    evaluated_count = len(evaluated_ngrams)

    overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
    overlapping_count = len(overlapping_ngrams)

    if evaluated_count == 0:
        precision = 0.0
    else:
        precision = overlapping_count / evaluated_count

    if reference_count == 0:
        recall = 0.0
    else:
        recall = overlapping_count / reference_count

    f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))
    return {"f": f1_score, "p": precision, "r": recall}


def get_rouge_score(cand_text, ref_text):
    """
    Compute ROUGE scores using the rouge_score package.
    :param cand_text:
    :param ref_text:
    :return: A dict of ROUGE scores
    """
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(cand_text, ref_text)
    return rouge_scores


def test_rouge(results_path):
    """
    Compute ROUGE scores using the rouge_score package.
    :param results_path: Path to store rouge score and summaries
    :return: (A dict of ROUGE scores,
              A dict of average ROUGE scores of LegalSum,
              A dict of average ROUGE scores of TOS;dr)
    """
    with open(results_path, "r+") as read_pred:
        try:
            curr_dict = json.load(read_pred)
        except json.decoder.JSONDecodeError:
            logger.error(".json file does not exist.")

    dict_keys = curr_dict.keys()
    avg_rouge_tosdr = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    tosdr_count = 0
    avg_rouge_legalsum = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    legalsum_count = 0
    with open(results_path, "w+") as save_pred:
        for key in dict_keys:
            pred_dict = curr_dict[key]
            candidate = pred_dict["extractive_summary"]
            candidate = candidate.replace("[CLS]", "").replace("[SEP]", "")
            reference = pred_dict["reference_summary"]

            rouge_results = get_rouge_score(candidate, reference)

            pred_dict["rouge_score"] = rouge_results
            key = str(pred_dict["uid"])
            curr_dict[key] = pred_dict

            rouge1 = rouge_results["rouge1"].precision
            rouge2 = rouge_results["rouge2"].precision
            rougeL = rouge_results["rougeL"].precision

            if "legalsum" in key:
                avg_rouge_legalsum["rouge1"] += rouge1
                avg_rouge_legalsum["rouge2"] += rouge2
                avg_rouge_legalsum["rougeL"] += rougeL
                legalsum_count += 1
            elif "tosdr" in key:
                avg_rouge_tosdr["rouge1"] += rouge1
                avg_rouge_tosdr["rouge2"] += rouge2
                avg_rouge_tosdr["rougeL"] += rougeL
                tosdr_count += 1

        avg_rouge_legalsum["rouge1"] /= legalsum_count
        avg_rouge_legalsum["rouge2"] /= legalsum_count
        avg_rouge_legalsum["rougeL"] /= legalsum_count
        avg_rouge_tosdr["rouge1"] /= tosdr_count
        avg_rouge_tosdr["rouge2"] /= tosdr_count
        avg_rouge_tosdr["rougeL"] /= tosdr_count

        json.dump(curr_dict, save_pred)

    print("Average ROUGE score of the LegalSum dataset is: ", avg_rouge_legalsum)
    print("Average ROUGE score of the TOS;dr dataset is: ", avg_rouge_tosdr)
    return curr_dict, avg_rouge_legalsum, avg_rouge_tosdr


# result_fp = "results/summary_distilbert.json"
# dir = result_fp.split("/")[0] + "/"
# if not os.path.exists(dir):
#     os.makedirs(dir)
# test_rouge(result_fp)

##################################################################################################
# class StepCheckpointCallback(pl.callbacks.base.Callback):
#     def __init__(
#         self, step_interval=1000, save_name="model", save_path=".", num_saves_to_keep=5
#     ):
#         super(StepCheckpointCallback, self).__init__()
#         self.step_interval = step_interval
#         self.save_name = save_name
#         self.save_path = save_path
#         self.num_saves_to_keep = num_saves_to_keep
#
#     def on_batch_end(self, trainer, pl_module):  # skipcq: PYL-W0613
#         # check if `step_interval` has passed and that the `global_step` is not 0
#         if (
#             trainer.global_step % self.step_interval == 0
#             and not trainer.global_step == 0
#         ):
#             logger.info(
#                 "Saving model to %s.ckpt at step %i.",
#                 self.save_path,
#                 trainer.global_step,
#             )
#             final_save_location = os.path.join(
#                 self.save_path,
#                 (self.save_name + "." + str(trainer.global_step) + ".ckpt"),
#             )
#             trainer.save_checkpoint(final_save_location)
#             # remove previous saves
#             offset = self.step_interval * self.num_saves_to_keep
#             path_to_remove = (
#                 self.save_name + "." + str(trainer.global_step - offset) + ".ckpt"
#             )
#             if os.path.isfile(path_to_remove):
#                 os.remove(path_to_remove)
#
#
# def lr_lambda_func(current_step, num_warmup_steps, num_training_steps):
#     if current_step < num_warmup_steps:
#         return float(current_step) / float(max(1, num_warmup_steps))
#     return max(
#         0.0,
#         float(num_training_steps - current_step)
#         / float(max(1, num_training_steps - num_warmup_steps)),
#     )
#
#
# def _get_word_ngrams(n, sentences):
#     """Calculates word n-grams for multiple sentences."""
#     assert len(sentences) > 0
#     assert n > 0
#
#     # words = _split_into_words(sentences)
#
#     words = sum(sentences, [])
#     # words = [w for w in words if w not in stopwords]
#     return _get_ngrams(n, words)
#
#
# def pad(data, pad_id, width=None, pad_on_left=False, nearest_multiple_of=False):
#     """
#     Pad ``data`` with ``pad_id`` to ``width`` on the right by default but if
#     ``pad_on_left`` then left.
#     """
#     if not width:
#         width = max(len(d) for d in data)
#     if nearest_multiple_of:
#         width = math.ceil(width / nearest_multiple_of) * nearest_multiple_of
#     if pad_on_left:
#         rtn_data = [[pad_id] * (width - len(d)) + d for d in data]
#     else:
#         rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
#     return rtn_data
#
#
# def pad_tensors(
#     tensors, pad_id=0, width=None, pad_on_left=False, nearest_multiple_of=False
# ):
#     """
#     Pad ``tensors`` with ``pad_id`` to ``width`` on the right by default but
#     if ``pad_on_left`` then left.
#     """
#     if not width:
#         width = max(len(d) for d in tensors)
#     if nearest_multiple_of:
#         width = math.ceil(width / nearest_multiple_of) * nearest_multiple_of
#     if pad_on_left:
#         return F.pad(
#             tensors,
#             pad=((width - tensors.size()[-1]), 0),
#             mode="constant",
#             value=pad_id,
#         )
#     return F.pad(
#         tensors,
#         pad=(0, (width - tensors.size()[-1])),
#         mode="constant",
#         value=pad_id,
#     )
#
#
# class LabelSmoothingLoss(nn.Module):
#     """
#     CrossEntropyLoss with label smoothing,
#     KL-divergence between q_{smoothed ground truth prob.}(w)
#     and p_{prob. computed by model}(w) is minimized.
#     From OpenNMT with modifications: https://github.com/OpenNMT/OpenNMT-py/blob/e8622eb5c6117269bb3accd8eb6f66282b5e67d9/onmt/utils/loss.py#L186
#     """  # noqa: E501
#
#     def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
#         assert 0.0 < label_smoothing <= 1.0
#         self.ignore_index = ignore_index
#         super(LabelSmoothingLoss, self).__init__()
#
#         smoothing_value = label_smoothing / (tgt_vocab_size - 2)
#         one_hot = torch.full((tgt_vocab_size,), smoothing_value)
#         one_hot[self.ignore_index] = 0
#         self.register_buffer("one_hot", one_hot.unsqueeze(0))
#
#         self.confidence = 1.0 - label_smoothing
#
#     def forward(self, output, target):
#         """
#         output (FloatTensor): batch_size x n_classes
#         target (LongTensor): batch_size
#         """
#         output = output.log_softmax(dim=1)
#
#         model_prob = self.one_hot.repeat(target.size(0), 1)
#         model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
#         model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)
#
#         return F.kl_div(output, model_prob, reduction="batchmean")
#
#
# # https://github.com/huggingface/transformers/blob/dc31a72f505bc115a2214a68c8ea7c956f98fd1b/examples/seq2seq/utils.py#L158
# def get_optimizer(hparams, optimizer_grouped_parameters):
#     if hparams.optimizer_type == "ranger":
#         optimizer = torch_optimizer.Ranger(
#             optimizer_grouped_parameters,
#             lr=hparams.learning_rate,
#             k=hparams.ranger_k,
#             eps=hparams.adam_epsilon,
#         )
#     elif hparams.optimizer_type == "qhadam":
#         optimizer = torch_optimizer.QHAdam(
#             optimizer_grouped_parameters,
#             lr=hparams.learning_rate,
#             nus=(0.1, 1.0),
#             betas=(0.9, 0.999),
#             eps=hparams.adam_epsilon,
#         )
#     elif hparams.optimizer_type == "radam":
#         optimizer = torch_optimizer.RAdam(
#             optimizer_grouped_parameters,
#             lr=hparams.learning_rate,
#             betas=(0.9, 0.999),
#             eps=hparams.adam_epsilon,
#         )
#     elif hparams.optimizer_type == "adabound":
#         optimizer = torch_optimizer.AdaBound(
#             optimizer_grouped_parameters,
#             lr=hparams.learning_rate,
#             betas=(0.9, 0.999),
#             final_lr=0.1,
#             gamma=1e-3,
#             eps=hparams.adam_epsilon,
#             amsbound=False,
#         )
#     else:
#         optimizer = torch.optim.AdamW(
#             optimizer_grouped_parameters,
#             lr=hparams.learning_rate,
#             eps=hparams.adam_epsilon,
#         )
#
#     return optimizer
#
#
# class SortishSampler(Sampler):
#     """
#     Go through the text data by order of src length with a bit of randomness.
#     From fastai repo with modifications.
#     """
#
#     def __init__(self, data, batch_size, pad_token_id):
#         self.data, self.bs, self.pad_token_id = data, batch_size, pad_token_id
#
#     def key(self, i):
#         current_item = self.data[int(i)]["source"]
#         return len(current_item[current_item != self.pad_token_id])
#
#     def __len__(self) -> int:
#         return len(self.data)
#
#     def __iter__(self):
#         idxs = np.random.permutation(len(self.data))
#         sz = self.bs * 50
#         ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
#         sort_idx = np.concatenate(
#             [sorted(s, key=self.key, reverse=True) for s in ck_idx]
#         )
#         sz = self.bs
#         ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
#         max_ck = np.argmax(
#             [self.key(ck[0]) for ck in ck_idx]
#         )  # find the chunk with the largest key,
#         ck_idx[0], ck_idx[max_ck] = (
#             ck_idx[max_ck],
#             ck_idx[0],
#         )  # then make sure it goes first.
#         sort_idx = (
#             np.concatenate(np.random.permutation(ck_idx[1:]))
#             if len(ck_idx) > 1
#             else np.array([], dtype=np.int)
#         )
#         sort_idx = np.concatenate((ck_idx[0], sort_idx)).tolist()
#         return iter(sort_idx)
#
#
# def generic_configure_optimizers(hparams, train_dataloader, params_to_update):
#     """
#     Configure the optimizers. Returns the optimizer and scheduler specified by
#     the values in ``hparams``. This is a generic function that both the extractive
#     and abstractive scripts use.
#     """
#     # check that max_steps is not None and is greater than 0
#     if hparams.max_steps and hparams.max_steps > 0:
#         # pytorch_lightning steps the scheduler every batch but only updates
#         # the global_step every gradient accumulation cycle. Therefore, the
#         # scheduler needs to have `accumulate_grad_batches` * `max_steps` in
#         # order to reach `max_steps`.
#         # See: https://github.com/PyTorchLightning/pytorch-lightning/blob/f293c9b5f4b4f9fabb2eec0c369f08a66c57ef14/pytorch_lightning/trainer/training_loop.py#L624  # noqa: E501
#         t_total = hparams.max_steps * hparams.accumulate_grad_batches
#     else:
#         t_total = int(
#             (
#                 len(train_dataloader.dataset)
#                 // (hparams.batch_size * max(1, hparams.gpus))
#             )
#             * hparams.max_epochs
#             // hparams.accumulate_grad_batches
#         )
#         if hparams.overfit_batches > 0.0:
#             t_total = int(t_total * hparams.overfit_batches)
#
#     no_decay = ["bias", "LayerNorm.weight"]
#     optimizer_grouped_parameters = [
#         {
#             "params": [
#                 p for n, p in params_to_update if not any(nd in n for nd in no_decay)
#             ],
#             "weight_decay": hparams.weight_decay,
#         },
#         {
#             "params": [
#                 p for n, p in params_to_update if any(nd in n for nd in no_decay)
#             ],
#             "weight_decay": 0.0,
#         },
#     ]
#
#     optimizer = get_optimizer(hparams, optimizer_grouped_parameters)
#
#     if hparams.use_scheduler:
#         if hparams.use_scheduler == "linear":
#             # We have to import the function and create a partial because functions cannot be
#             # serialized by python pickle. Therefore, if the normal
#             # `get_linear_schedule_with_warmup` function provided by `transformers` was used,
#             # the program would fail to save `hparams` because the optimizer would contain a
#             # locale function that cannot be pickled.
#             lr_lambda = partial(
#                 lr_lambda_func,
#                 num_warmup_steps=hparams.warmup_steps * hparams.accumulate_grad_batches,
#                 num_training_steps=t_total,
#             )
#             # multiply by `hparams.accumulate_grad_batches` above because pytorch_lightning
#             # steps are for each batch, except for the `trainer.global_step`, which tracks
#             # the actual number of steps
#
#             scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, -1)
#
#         elif hparams.use_scheduler == "onecycle":
#             scheduler = torch.optim.lr_scheduler.OneCycleLR(
#                 optimizer, max_lr=hparams.learning_rate, total_steps=t_total
#             )
#         elif hparams.use_scheduler == "poly":
#             from poly_lr_decay import PolynomialLRDecay
#
#             scheduler = PolynomialLRDecay(
#                 optimizer,
#                 end_learning_rate=hparams.end_learning_rate,
#                 max_decay_steps=t_total,
#                 power=3.0,
#             )
#         else:
#             logger.error(
#                 "The value %s for `--use_scheduler` is invalid.",
#                 hparams.use_scheduler,
#             )
#         # the below interval is called "step" but the scheduler is moved forward
#         # every batch.
#         scheduler_dict = {"scheduler": scheduler, "interval": "step"}
#
#         return ([optimizer], [scheduler_dict])
#
#     return optimizer
#
#
# def strip_extra_spaces_and_newline(text):
#     text = re.sub(r"\n", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text

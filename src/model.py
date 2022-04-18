import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (BertConfig, BertModel, DistilBertConfig,
                          DistilBertModel)

from encoder import ExtTransformerEncoder

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
BERT_BASE_CHECKPOINT_NAME = CHECKPOINT_DIR / "bertbase" / "bertbase_checkpoint"
ALTERNATE_CHECKPOINT_NAME = CHECKPOINT_DIR / "alternate" / "alternate_checkpoint"

if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir()

for d in [BERT_BASE_CHECKPOINT_NAME, ALTERNATE_CHECKPOINT_NAME]:
    if not d.parent.exists():
        d.parent.mkdir()


class Bert(nn.Module):
    def __init__(self, bert_type="bertbase"):
        super(Bert, self).__init__()
        self.bert_type = bert_type

        if bert_type == "bertbase":
            configuration = BertConfig()
            self.model = BertModel(configuration)
        elif bert_type == "distilbert":
            configuration = DistilBertConfig()
            self.model = DistilBertModel(configuration)

    def forward(self, x, segs, mask):
        if self.bert_type == "distilbert":
            top_vec = self.model(input_ids=x, attention_mask=mask)[0]
        else:
            top_vec, _ = self.model(x, attention_mask=mask, token_type_ids=segs)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, device, checkpoint=None, bert_type="bertbase"):
        super().__init__()
        self.device = device
        self.bert = Bert(bert_type=bert_type)
        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size,
            d_ff=2048,
            heads=8,
            dropout=0.2,
            num_inter_layers=2,
        )

        if checkpoint is not None:
            self.load_state_dict(checkpoint, strict=True)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls


def get_extractive_summarizer(model_type: str = "distilbert", device="cuda"):
    if device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        print("CUDA not available, using CPU.")
        device = torch.device("cpu")
    if model_type == "bertbase":
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
        if not ALTERNATE_CHECKPOINT_NAME.exists():
            os.system(
                f'curl "https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE" -o {str(ALTERNATE_CHECKPOINT_NAME)}'
            )
        checkpoint = torch.load(
            # f"1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE",
            str(ALTERNATE_CHECKPOINT_NAME),
            map_location=device,
        )
    model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device=device)
    return model

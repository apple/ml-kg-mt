#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
Knowledge Graph + Machine Translation (KG-MT) model.

This module provides a class to train a model that translates text by using a knowledge
retriever to select relevant entities from a knowledge graph and augment the input with
the retrieved knowledge. The model then uses a sequence-to-sequence transformer to
translate the augmented input to the target language.

The module also provides functions to load datasets from local and remote sources, as
well as a script to train and evaluate the model.
"""

import json
import random
import re
import sqlite3
from typing import Any, Dict, List

import datasets as ds
import lightning.pytorch as pl
import stanza
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from sacrebleu.metrics import BLEU
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.contriever import Contriever

KG_PATH = "data/wikidata/entities.wikidata.db"
RETRIEVER_MODEL_NAME = "checkpoints/retriever"
TRANSLATOR_MODEL_NAME = "nllb"
SPECIAL_TOKENS = ["<meta>", "<translated-as>"]

SUPPORTED_MODELS = {
    "nllb": "facebook/nllb-200-distilled-600M",
    "mbart": "facebook/mbart-large-50-many-to-many-mmt",
    "m2m": "facebook/m2m100_418M",
}

NLLB_LANGUAGE_MAP = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "th": "tha_Thai",
    "tr": "tur_Latn",
    "zh": "zho_Hant",
}
NLLB_INVERSE_LANGUAGE_MAP = {v: k for k, v in NLLB_LANGUAGE_MAP.items()}

MBART_LANGUAGE_MAP = {
    "en": "en_XX",
    "ar": "ar_AR",
    "de": "de_DE",
    "es": "es_XX",
    "fr": "fr_XX",
    "it": "it_IT",
    "ja": "ja_XX",
    "ko": "ko_KR",
    "th": "th_TH",
    "tr": "tr_TR",
    "zh": "zh_CN",
}
MBART_INVERSE_LANGUAGE_MAP = {v: k for k, v in MBART_LANGUAGE_MAP.items()}

M2M_LANGUAGE_MAP = {
    "ar": "ar",
    "de": "de",
    "en": "en",
    "fr": "fr",
    "es": "es",
    "it": "it",
    "ja": "ja",
    "ko": "ko",
    "th": "th",
    "tr": "tr",
    "zh": "zh",
}
M2M_INVERSE_LANGUAGE_MAP = {v: k for k, v in M2M_LANGUAGE_MAP.items()}

LANGUAGE_MAP = {
    "nllb": NLLB_LANGUAGE_MAP,
    "mbart": MBART_LANGUAGE_MAP,
    "m2m": M2M_LANGUAGE_MAP,
}

INVERSE_LANGUAGE_MAP = {
    "nllb": NLLB_INVERSE_LANGUAGE_MAP,
    "mbart": MBART_INVERSE_LANGUAGE_MAP,
    "m2m": M2M_INVERSE_LANGUAGE_MAP,
}

MENTIONS_PARAMS = {
    "ar": [3, 10],
    "en": [3, 10],
    "es": [3, 10],
    "de": [3, 10],
    "fr": [3, 10],
    "it": [3, 10],
    "ja": [2, 10],
    "ko": [2, 10],
    "th": [3, 10],
    "tr": [3, 10],
    "zh": [2, 10],
}

DEFAULT_TRAIN_BATCH_SIZE = 1
DEFAULT_VAL_BATCH_SIZE = 8
DEFAULT_TEST_BATCH_SIZE = 32
DEFAULT_PRED_BATCH_SIZE = 32
DEFAULT_LIMIT_VAL_BATCHES = 10
DEFAULT_BATCH_ACCUMULATION = 4
DEFAULT_VALIDATION_CHECK_INTERVAL = DEFAULT_BATCH_ACCUMULATION * 500
DEFAULT_MAX_STEPS = 100_000
DEFAULT_MAX_EPOCHS = 10
DEFAULT_PATIENCE = 10
DEFAULT_GRADIENT_CLIP_VALUE = 0.5


class KgMtBatch:
    def __init__(self, data: List[dict]):
        self.size = len(data)
        self.instance_ids = []
        self.src_text = []
        self.tgt_text = []
        self.src_lang = []
        self.tgt_lang = []
        self.candidate_entities = {
            "wikidata_ids": [],
            "descriptions": [],
            "source_names": [],
            "target_names": [],
            "entity_indices": [],
        }
        entity_index = 0

        for instance in data:
            self.instance_ids.append(instance["id"])
            self.src_text.append(instance["source"])
            self.src_lang.append(instance["src_lang"])
            self.tgt_lang.append(instance["tgt_lang"])
            if "target" in instance:
                self.tgt_text.append(instance["target"])

            if instance["wikidata_ids"]:
                _wikidata_ids = instance["wikidata_ids"]
                _wikidata_descriptions = instance["wikidata_descriptions"]
                _source_names = [n[0] for n in instance["wikidata_names"]]
                _target_names = [n[1] for n in instance["wikidata_names"]]
            else:
                _wikidata_ids = []
                _wikidata_descriptions = []
                _source_names = []
                _target_names = []

            self.candidate_entities["wikidata_ids"].append(_wikidata_ids)
            self.candidate_entities["descriptions"].extend(_wikidata_descriptions)
            self.candidate_entities["source_names"].append(_source_names)
            self.candidate_entities["target_names"].append(_target_names)

            _entity_indices = [entity_index + i for i in range(len(_wikidata_ids))]
            self.candidate_entities["entity_indices"].append(_entity_indices)
            entity_index += len(_wikidata_ids)

    @staticmethod
    def collate_fn(instances: List[dict]):
        return KgMtBatch(instances)


class KgMt(pl.LightningModule):
    def __init__(
        self,
        log_info: Dict[int, dict] = None,
        retrieval_model_name: str = RETRIEVER_MODEL_NAME,
        translation_model_name: str = TRANSLATOR_MODEL_NAME,
        special_tokens: List[str] = SPECIAL_TOKENS,
        num_beams: int = 1,
    ):
        super().__init__()
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retrieval_model_name)
        self.retriever_model = Contriever.from_pretrained(retrieval_model_name).eval()

        self.language_map = LANGUAGE_MAP[translation_model_name]
        self.inverse_language_map = INVERSE_LANGUAGE_MAP[translation_model_name]

        translation_model_name = SUPPORTED_MODELS[translation_model_name]
        self.translator_tokenizer = AutoTokenizer.from_pretrained(
            translation_model_name,
            src_lang=self.language_map["en"],
            tgt_lang=self.language_map["en"],
        )
        self.translator_tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name)
        self.translator.resize_token_embeddings(len(self.translator_tokenizer))

        self.retriever_projection = torch.nn.Linear(
            self.retriever_model.encoder.config.hidden_size,
            self.translator.model.encoder.config.hidden_size,
            bias=False,
        )

        self.pre_layer_norm = torch.nn.LayerNorm(
            self.retriever_model.encoder.config.hidden_size
        )
        self.post_layer_norm = torch.nn.LayerNorm(
            self.translator.model.encoder.config.hidden_size
        )

        self.warmup_steps = 500
        self.max_retrieved_entities = 3
        self.num_beams = num_beams
        self.log_info = log_info
        self.val_step_outputs = []
        self.test_step_outputs = []

    def select_relevant_entities(
        self,
        source_text: List[str],
        candidate_entities: Dict[str, list],
    ) -> List[dict]:
        # Tokenize source text and source descriptions.
        retriever_inputs = self.retriever_tokenizer(
            text=source_text + candidate_entities["descriptions"],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Get the embeddings for the source texts and descriptions.
        embeddings = self.retriever_model(**retriever_inputs)
        source_text_embeddings = embeddings[: len(source_text)]
        candidate_embeddings = embeddings[len(source_text) :]

        # Obtain the most relevant entities and their embeddings for each source text.
        retrieved_entities = []
        retrieved_source_names = []
        retrieved_target_names = []
        retrieved_embeddings = []

        for i in range(len(source_text)):
            entity_indices = candidate_entities["entity_indices"][i]
            entity_ids = candidate_entities["wikidata_ids"][i]
            if not entity_ids:
                retrieved_entities.append([])
                retrieved_source_names.append([])
                retrieved_target_names.append([])
                retrieved_embeddings.append(source_text_embeddings[i])
                continue

            # Compute similarity between source text and entities.
            source_text_embedding = source_text_embeddings[i]
            entity_embeddings = candidate_embeddings[entity_indices]
            similarities = F.cosine_similarity(
                source_text_embedding,
                entity_embeddings,
            ).tolist()

            # Remove entities with low scores.
            entity_info = [
                (w, s, e)
                for w, s, e in zip(entity_ids, similarities, entity_embeddings)
                if s >= 0.5
            ]

            if not entity_info:
                retrieved_entities.append([])
                retrieved_source_names.append([])
                retrieved_target_names.append([])
                retrieved_embeddings.append(source_text_embedding)
                continue

            # Sort by similarity (descending).
            entity_info = sorted(
                entity_info,
                key=lambda x: x[1],
                reverse=True,
            )

            # Get retrieved entity IDs.
            relevant_entities = {w: e for w, _, e in entity_info}

            # Get source and target entity names for retrieved entities.
            _retrieved_entities = []
            _retrieved_source_names = []
            _retrieved_target_names = []
            _retrieved_embeddings = [source_text_embedding]

            for entity_idx, entity_id in enumerate(entity_ids):
                if entity_id in relevant_entities:
                    source_name = candidate_entities["source_names"][i][entity_idx]
                    target_name = candidate_entities["target_names"][i][entity_idx]

                    if source_name.casefold() != target_name.casefold():
                        _retrieved_entities.append(entity_id)
                        _retrieved_source_names.append(source_name)
                        _retrieved_target_names.append(target_name)
                        _retrieved_embeddings.append(relevant_entities[entity_id])

            retrieved_entities.append(
                _retrieved_entities[: self.max_retrieved_entities]
            )
            retrieved_source_names.append(
                _retrieved_source_names[: self.max_retrieved_entities]
            )
            retrieved_target_names.append(
                _retrieved_target_names[: self.max_retrieved_entities]
            )

            # Get vector representation of retrieved entities.
            retrieved_embedding = torch.mean(
                torch.stack(_retrieved_embeddings[: self.max_retrieved_entities + 1]),
                dim=0,
            ).to(self.device)
            retrieved_embeddings.append(retrieved_embedding)

        return {
            "entities": retrieved_entities,
            "source_names": retrieved_source_names,
            "target_names": retrieved_target_names,
            "embeddings": retrieved_embeddings,
        }

    def build_knowledge_metadata(
        self,
        source_names: List[List[str]],
        target_names: List[List[str]],
    ) -> List[str]:
        metadata = []

        for entity_source_names, entity_target_names in zip(source_names, target_names):
            entity_metadata = []

            # Combine every source and target names (e.g., "source -> target").
            for source, target in zip(entity_source_names, entity_target_names):
                entity_metadata.append(f"{source} <translated-as> {target}")

            # Join in a single string (e.g., "s1 = t1; s2 = t2")
            entity_metadata = "<meta> " + "<meta> ".join(entity_metadata)
            metadata.append(entity_metadata)

        return metadata

    def augment_inputs_with_knowledge(
        self,
        source_text: List[str],
        source_locale: List[str],
        knowledge_metadata: List[str],
    ) -> List[str]:
        augmented_inputs = []

        for text, locale, data in zip(source_text, source_locale, knowledge_metadata):
            # Add the metadata at the end of the string.
            if data:
                augmented_text = f"{locale} {text} {data}</s>"
            else:
                augmented_text = f"{locale} {text}</s>"
            augmented_inputs.append(augmented_text)

        return augmented_inputs

    def forward(self, batch: KgMtBatch, generate: bool = False):
        # Get most relevant entities using the knowledge retriever.
        # Returns Wikidata IDs, entity embeddings, source primary names, target primary names.
        with torch.no_grad():
            relevant_entities = self.select_relevant_entities(
                batch.src_text, batch.candidate_entities
            )

        # Build metadata starting from source and target primary names.
        knowledge_metadata = self.build_knowledge_metadata(
            relevant_entities["source_names"], relevant_entities["target_names"]
        )

        # Augment input text with retrieved knowledge.
        knowledge_augmented_inputs = self.augment_inputs_with_knowledge(
            batch.src_text, batch.src_lang, knowledge_metadata
        )

        # Tokenize input text.
        mt_inputs = self.translator_tokenizer(
            text=knowledge_augmented_inputs,
            text_target=[t[0] for t in batch.tgt_text] if batch.tgt_text else None,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Project knowledge embedding to have the same size as the translator embeddings.
        knowledge_embedding = self.retriever_projection(
            self.pre_layer_norm(torch.stack(relevant_entities["embeddings"])).unsqueeze(
                1
            )
        )

        # Prepend knowledge embedding to input_embeddings.
        input_embeddings = (
            torch.concat(
                [
                    self.post_layer_norm(knowledge_embedding),
                    self.translator.model.encoder.embed_tokens(mt_inputs.input_ids),
                ],
                dim=1,
            )
            * self.translator.model.encoder.embed_scale
        )

        # Extend attention_mask to have the same sequence length as input_embeddings.
        attention_mask = torch.concat(
            [
                torch.ones(mt_inputs.attention_mask.shape[0], 1, device=self.device),
                mt_inputs.attention_mask,
            ],
            dim=-1,
        )

        # Compute translation outputs.
        if "labels" in mt_inputs and not generate:
            mt_outputs = self.translator(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=mt_inputs.labels,
            )

        else:
            # Prepare decoder_input_ids.
            decoder_input_ids = torch.tensor(
                [[self.translator_tokenizer.lang_code_to_id[l]] for l in batch.tgt_lang]
            ).to(self.device)

            mt_outputs = self.translator.generate(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=128,
                num_beams=self.num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=0.5,
            )

        mt_outputs.entities = relevant_entities["entities"]
        mt_outputs.source_names = relevant_entities["source_names"]
        mt_outputs.target_names = relevant_entities["target_names"]

        return mt_outputs

    def training_step(self, batch: KgMtBatch, batch_idx: int) -> float:
        outputs = self(batch)
        loss = outputs["loss"]
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch.size,
        )
        return loss

    def validation_step(
        self,
        batch: KgMtBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        eval_output = self._shared_eval_step(batch, dataloader_idx, "val")
        self.val_step_outputs.append(eval_output)

    def test_step(
        self,
        batch: KgMtBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        eval_output = self._shared_eval_step(batch, dataloader_idx, "test")
        self.test_step_outputs.append(eval_output)

    def predict_step(
        self,
        batch: KgMtBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> List[str]:
        model_outputs = self(batch, generate=True)
        batch.predictions = self.translator_tokenizer.batch_decode(
            model_outputs,
            skip_special_tokens=True,
        )
        batch.relevant_entities = model_outputs.entities
        batch.relevant_source_names = model_outputs.source_names
        batch.relevant_target_names = model_outputs.target_names
        return dataloader_idx, batch

    def _shared_eval_step(
        self,
        batch: KgMtBatch,
        dataloader_idx: int,
        stage: str,
    ) -> dict:
        outputs = self(batch)
        loss = outputs["loss"]

        if self.log_info and dataloader_idx in self.log_info:
            dataset_name = self.log_info[dataloader_idx]["dataset_name"]
            src_lang = self.inverse_language_map[
                self.log_info[dataloader_idx]["src_lang"]
            ]
            tgt_lang = self.inverse_language_map[
                self.log_info[dataloader_idx]["tgt_lang"]
            ]
            self.log(
                f"{dataset_name}/{stage}_loss/{src_lang}-{tgt_lang}",
                loss,
                on_step=True,
                on_epoch=True,
                batch_size=batch.size,
                add_dataloader_idx=False,
            )

        return {
            "loss": loss,
            "batch": batch,
            "dataloader_idx": dataloader_idx,
        }

    def on_validation_epoch_end(self):
        self._shared_epoch_end(self.val_step_outputs, "val")

    def on_test_epoch_end(self):
        self._shared_epoch_end(self.test_step_outputs, "test")

    def _shared_epoch_end(self, step_outputs: List[dict], stage: str):
        loss = torch.mean(torch.stack([x["loss"] for x in step_outputs]))
        self.log(f"{stage}_loss", loss, prog_bar=True)

        results = {}
        for sample in step_outputs:
            dataloader_idx: int = sample["dataloader_idx"]
            src_lang: str = self.log_info[dataloader_idx]["src_lang"]
            tgt_lang: str = self.log_info[dataloader_idx]["tgt_lang"]
            batch: KgMtBatch = sample["batch"]
            language_pair = (
                self.inverse_language_map[src_lang],
                self.inverse_language_map[tgt_lang],
            )

            if language_pair not in results:
                results[language_pair] = {"references": [], "predictions": []}
            results[language_pair]["references"].extend(batch.tgt_text)

            predicted_ids = self(batch, generate=True)
            predictions = self.translator_tokenizer.batch_decode(
                predicted_ids, skip_special_tokens=True
            )

            results[language_pair]["predictions"].extend(predictions)

        avg_bleu_score = 0.0
        for (src_lang, tgt_lang), corpus in results.items():
            bleu = BLEU(trg_lang=tgt_lang)
            predictions = corpus["predictions"]
            references = list(map(list, zip(*corpus["references"])))
            bleu_score = bleu.corpus_score(predictions, references)
            self.log(f"val_bleu/{src_lang}-{tgt_lang}", bleu_score.score)
            avg_bleu_score += bleu_score.score

        avg_bleu_score /= len(results)
        self.log(f"{stage}_bleu", avg_bleu_score, prog_bar=True)

        step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=3e-5,
            eps=1e-6,
            betas=(0.9, 0.98),
        )
        scheduler = transformers.get_inverse_sqrt_schedule(optimizer, self.warmup_steps)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        ]


def _map_local_dataset_columns(
    instance: Dict[str, Any],
    src_lang: str,
    tgt_lang: str,
    swap: bool,
):
    if not swap:
        source = instance["source"]
        target = [instance["target"][0]]
    else:
        source = instance["target"][0]
        target = [instance["source"]]

    return {
        "id": "{}-{}-{}".format(instance.get("instance_id", -1), src_lang, tgt_lang),
        "source": source,
        "target": target,
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }


def _map_remote_dataset_columns(
    instance: Dict[str, Any],
    idx: int,
    data_src_lang: str,
    data_tgt_lang: str,
    src_lang: str,
    tgt_lang: str,
):
    return {
        "id": f"nllb-{src_lang}-{tgt_lang}_{idx}",
        "source": instance["translation"][data_src_lang],
        "target": [instance["translation"][data_tgt_lang]],
        "src_lang": src_lang,
        "tgt_lang": tgt_lang,
    }


def _generate_mentions(
    text: str,
    tokenizer: stanza.Pipeline,
    min_unigram_length: int,
    max_ngram_size: int,
) -> List[str]:
    sentence = tokenizer(text.casefold()).sentences[0]
    tokens = [t.text for t in sentence.tokens]
    ngrams = set(ngram for ngram in tokens if len(ngram) > min_unigram_length)

    for ngram_size in range(2, max_ngram_size + 1):
        current_ngrams = zip(*[tokens[i:] for i in range(ngram_size)])

        for ngram in current_ngrams:
            ngram = " ".join(ngram)
            ngrams.add(ngram)
    return list(ngrams)


def _retrieve_candidates(
    instance: Dict[str, Any],
    con: sqlite3.Connection,
    src_lang: str,
    tgt_lang: str,
    tokenizer: stanza.Pipeline,
    mode: str = "gold",
    gold_probability: float = 1.0,
    skip_probability: float = 0.0,
):
    candidates = {
        "wikidata_ids": None,
        "wikidata_names": None,
        "wikidata_descriptions": None,
    }

    if mode == "skip" or random.random() < skip_probability:
        return candidates

    elif (
        mode == "gold"
        and random.random() < gold_probability
        and ("entities" in instance or "wikidata_id" in instance)
    ):
        wikidata_ids = (
            instance["entities"]
            if "entities" in instance
            else [instance["wikidata_id"]]
        )

    else:
        source = instance["source"]
        mentions = _generate_mentions(
            source,
            tokenizer,
            MENTIONS_PARAMS[src_lang][0],
            MENTIONS_PARAMS[src_lang][1],
        )
        wikidata_ids = get_wikidata_ids(con, mentions, src_lang)

    wikidata_src_info = get_wikidata_info(con, wikidata_ids, src_lang)
    wikidata_tgt_info = get_wikidata_info(con, wikidata_ids, tgt_lang)

    source_names = {w: n for w, n, _, _ in wikidata_src_info}
    target_names = {w: n for w, n, _, _ in wikidata_tgt_info}
    descriptions = {
        w: f"{n}: {d}" if d else n
        for w, n, d, _ in wikidata_src_info
        if w in target_names
    }

    wikidata_ids = [w for w in wikidata_ids if w in descriptions][:10]

    unique_names = set()
    wikidata_names = {}
    filtered_wikidata_ids = []

    for wikidata_id in wikidata_ids:
        source_name = re.sub(r" ?\(.*?\)", "", source_names[wikidata_id]).strip()
        target_name = re.sub(r" ?\(.*?\)", "", target_names[wikidata_id]).strip()
        normalized_name = source_name.lower()

        if normalized_name not in unique_names:
            wikidata_names[wikidata_id] = (source_name, target_name)
            filtered_wikidata_ids.append(wikidata_id)
            unique_names.add(normalized_name)

    if filtered_wikidata_ids and wikidata_names and descriptions:
        candidates = {
            "wikidata_ids": filtered_wikidata_ids,
            "wikidata_names": [wikidata_names[w] for w in filtered_wikidata_ids],
            "wikidata_descriptions": [descriptions[w] for w in filtered_wikidata_ids],
        }

    return candidates


def get_wikidata_ids(
    con: sqlite3.Connection,
    names: List[str],
    locale: str,
) -> List[str]:
    sql_get_objects = """
        SELECT DISTINCT wikidata_id
        FROM entity_names
        WHERE locale='{}' AND entity_name in ({});
    """.format(
        locale, ", ".join("?" for _ in names)
    )
    cur = con.cursor()
    cur.execute(sql_get_objects, names)
    return [w[0] for w in cur.fetchall()]


def get_wikidata_info(
    con: sqlite3.Connection,
    wikidata_ids: List[str],
    locale: str,
) -> List[str]:
    sql_get_objects = """
        SELECT wikidata_id, entity_name, entity_description, entity_popularity
        FROM entity_info
        WHERE locale='{}' AND wikidata_id in ({})
        ORDER BY entity_popularity DESC;
    """.format(
        locale, ", ".join("?" for _ in wikidata_ids)
    )
    cur = con.cursor()
    cur.execute(sql_get_objects, wikidata_ids)
    return cur.fetchall()


def load_local_datasets(
    data_paths: List[str],
    language_pairs: List[str],
    language_map: Dict[str, str],
    kg_path: str,
    datasets: List[ds.Dataset] = [],
    data_info: Dict[int, str] = {},
    split: str = "train",
    limit_samples: int = None,
    include_swapped_languages: bool = False,
    include_augmented_data_only: bool = False,
    mode: str = "gold",
    skip_probability: float = 0.0,
    gold_probability: float = 1.0,
):
    source_languages = set(
        [language for pair in language_pairs for language in pair.split("-")]
    )
    tokenizers = {
        l: stanza.Pipeline(
            lang=l, processors="tokenize", tokenize_no_ssplit=True, verbose=False
        )
        for l in source_languages
    }

    logger.info("Loading local data...")
    num_language_pairs = len(language_pairs)
    swap_languages = [False for _ in range(num_language_pairs)]

    if include_swapped_languages:
        data_paths = data_paths + data_paths
        language_pairs = language_pairs + language_pairs
        swap_languages = swap_languages + [True for _ in range(num_language_pairs)]

    for data_path, language_pair, swap in zip(
        data_paths, language_pairs, swap_languages
    ):
        con = sqlite3.connect(kg_path)
        src_lang, tgt_lang = language_pair.split("-")
        if swap:
            src_lang, tgt_lang = tgt_lang, src_lang

        model_src_lang = language_map[src_lang]
        model_tgt_lang = language_map[tgt_lang]

        logger.info(f"Loading data from {data_path}...")
        dataset = ds.load_dataset(
            "json",
            data_files=data_path,
            split=split,
            streaming=True,
        )
        dataset = dataset.filter(lambda x: x["target"])
        dataset = dataset.map(
            _map_local_dataset_columns,
            fn_kwargs={
                "src_lang": model_src_lang,
                "tgt_lang": model_tgt_lang,
                "swap": swap,
            },
        )
        dataset = dataset.map(
            _retrieve_candidates,
            fn_kwargs={
                "con": con,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "tokenizer": tokenizers[src_lang],
                "mode": mode,
                "skip_probability": skip_probability,
                "gold_probability": gold_probability,
            },
        )
        dataset = dataset.remove_columns(["source_locale", "target_locale"])

        if include_augmented_data_only:
            dataset = dataset.filter(lambda x: x["wikidata_ids"])
        if limit_samples:
            dataset = dataset.take(limit_samples)

        datasets.append(dataset)

        data_info[len(data_info)] = {
            "dataset_name": language_pair,
            "src_lang": model_src_lang,
            "tgt_lang": model_tgt_lang,
        }

    logger.info(f"Local datasets ready!")

    return datasets, data_info


def load_remote_datasets(
    language_pairs: List[str],
    language_map: Dict[str, str],
    kg_path: str,
    datasets: List[ds.Dataset] = [],
    data_info: Dict[int, str] = {},
    split: str = "train",
    include_swapped_languages: bool = False,
    limit_samples: int = None,
    skip_probability: float = 0.85,
):
    source_languages = set(
        [language for pair in language_pairs for language in pair.split("-")]
    )
    tokenizers = {
        l: stanza.Pipeline(
            lang=l, processors="tokenize", tokenize_no_ssplit=True, verbose=False
        )
        for l in source_languages
    }

    logger.info("Loading remote data...")
    num_language_pairs = len(language_pairs)
    swap_languages = [False for _ in range(num_language_pairs)]
    if include_swapped_languages:
        language_pairs = language_pairs + language_pairs
        swap_languages = swap_languages + [True for _ in range(num_language_pairs)]

    for language_pair, swap in zip(language_pairs, swap_languages):
        con = sqlite3.connect(kg_path)

        logger.info(f"Loading data for {language_pair}...")
        src_lang, tgt_lang = language_pair.split("-")
        nllb_src_lang = NLLB_LANGUAGE_MAP[src_lang]
        nllb_tgt_lang = NLLB_LANGUAGE_MAP[tgt_lang]
        language_pair = f"{nllb_src_lang}-{nllb_tgt_lang}"

        dataset = ds.load_dataset(
            "allenai/nllb",
            language_pair,
            split=split,
            streaming=True,
        )

        if swap:
            src_lang, tgt_lang = tgt_lang, src_lang
            nllb_src_lang, nllb_tgt_lang = nllb_tgt_lang, nllb_src_lang

        model_src_lang = language_map[src_lang]
        model_tgt_lang = language_map[tgt_lang]

        dataset = dataset.map(
            _map_remote_dataset_columns,
            with_indices=True,
            fn_kwargs={
                "data_src_lang": nllb_src_lang,
                "data_tgt_lang": nllb_tgt_lang,
                "src_lang": model_src_lang,
                "tgt_lang": model_tgt_lang,
            },
        )
        dataset = dataset.map(
            _retrieve_candidates,
            fn_kwargs={
                "con": con,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "tokenizer": tokenizers[src_lang],
                "mode": "auto",
                "skip_probability": skip_probability,
                "gold_probability": 0.0,
            },
        )
        dataset = dataset.remove_columns(["translation", "laser_score"])

        if limit_samples:
            dataset = dataset.take(limit_samples)

        datasets.append(dataset)

        data_info[len(data_info)] = {
            "dataset_name": f"nllb_{nllb_src_lang}-{nllb_tgt_lang}",
            "src_lang": model_src_lang,
            "tgt_lang": model_tgt_lang,
        }

    logger.info(f"Remote datasets ready!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        default="nllb",
        choices=list(SUPPORTED_MODELS.keys()),
        help="Name of the pretrained language model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--train_data_paths",
        type=str,
        nargs="+",
        required=False,
        help="Path(s) to the training datasets.",
    )
    parser.add_argument(
        "--val_data_paths",
        type=str,
        nargs="+",
        required=False,
        help="Path(s) to the validation datasets.",
    )
    parser.add_argument(
        "--test_data_paths",
        type=str,
        nargs="+",
        required=False,
        help="Path(s) to the test datasets.",
    )
    parser.add_argument(
        "--pred_input_paths",
        type=str,
        nargs="+",
        required=False,
        help="List of paths to the input files for prediction.",
    )
    parser.add_argument(
        "--pred_output_paths",
        type=str,
        nargs="+",
        required=False,
        help="List of output paths to the predictions when --predict is set.",
    )
    parser.add_argument(
        "--language_pairs",
        type=str,
        nargs="+",
        required=False,
        help="Language pairs of the data, e.g., en-es.",
    )
    parser.add_argument(
        "--remote_language_pairs",
        type=str,
        nargs="+",
        required=False,
        help="Language pairs of the remote data, e.g., en-es.",
    )
    parser.add_argument(
        "--include_augmented_data_only",
        action="store_true",
        help="Include only augmented data in local datasets.",
    )
    parser.add_argument(
        "--kg_path",
        type=str,
        required=True,
        help="Path to the KG (sqlite database).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=DEFAULT_TRAIN_BATCH_SIZE,
        help="Number of sentences per batch during training.",
    )
    parser.add_argument(
        "--val_batch_size",
        type=int,
        default=DEFAULT_VAL_BATCH_SIZE,
        help="Number of sentences per batch during validation.",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=DEFAULT_TEST_BATCH_SIZE,
        help="Number of sentences per batch during test.",
    )
    parser.add_argument(
        "--pred_batch_size",
        type=int,
        default=DEFAULT_PRED_BATCH_SIZE,
        help="Number of sentences per batch during prediction.",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=DEFAULT_LIMIT_VAL_BATCHES,
        help="Limit the number of validation batches for each dataloader.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=DEFAULT_BATCH_ACCUMULATION,
        help="Number of batches for gradient accumulation before backpropagation.",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=int,
        default=DEFAULT_GRADIENT_CLIP_VALUE,
        help="Value for gradient clipping.",
    )
    parser.add_argument(
        "--val_check_interval",
        type=int,
        default=DEFAULT_VALIDATION_CHECK_INTERVAL,
        help="Number of steps before validation.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of training steps.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=DEFAULT_MAX_EPOCHS,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Patience level.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Set this flag to train the model.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Set this flag to test the model.",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Set this flag to predict with the model.",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=1,
        help="Number of beams (beam size) for generation.",
    )
    parser.add_argument(
        "--use_gold_entities",
        action="store_true",
        help="Set this flag to use gold entities for prediction.",
    )
    parser.add_argument(
        "--disable_retriever",
        action="store_true",
        help="Set this flag to disable the knowledge retriever.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        required=False,
        default=1,
        help="GPUs to use, e.g., '1,3'.",
    )

    args = parser.parse_args()
    translation_model_name: str = args.model_name
    checkpoint_path: str = args.checkpoint
    kg_path: str = args.kg_path
    limit_val_batches: float = args.limit_val_batches
    accumulate_grad_batches: int = args.accumulate_grad_batches
    gradient_clip_val: int = args.gradient_clip_val
    val_check_interval: int = args.val_check_interval
    max_steps: int = args.max_steps
    max_epochs: int = args.max_epochs
    patience: int = args.patience
    num_beams: int = args.num_beams
    wandb_logger = None

    if args.train:
        train_data_paths: List[str] = args.train_data_paths
        val_data_paths: List[str] = args.val_data_paths
        language_pairs: List[str] = args.language_pairs
        remote_language_pairs: List[str] = args.remote_language_pairs
        include_augmented_data_only: bool = args.include_augmented_data_only
        train_batch_size: int = args.train_batch_size
        val_batch_size: int = args.val_batch_size
        assert train_data_paths, "--train_data_paths must be specified for --train"
        assert val_data_paths, "--val_data_paths must be specified for --train"
        assert language_pairs, "--language_pairs must be specified for --train"

        train_datasets = []

        load_local_datasets(
            train_data_paths,
            language_pairs,
            LANGUAGE_MAP[translation_model_name],
            kg_path,
            datasets=train_datasets,
            include_swapped_languages=True,
            include_augmented_data_only=include_augmented_data_only,
            gold_probability=0.9,
        )
        probabilities = [1.0 / (2 * len(language_pairs))] * (2 * len(language_pairs))

        if remote_language_pairs:
            load_remote_datasets(
                remote_language_pairs,
                LANGUAGE_MAP[translation_model_name],
                kg_path,
                datasets=train_datasets,
                include_swapped_languages=True,
                limit_samples=100_000,
                skip_probability=1.0,
            )
            probabilities = [0.5 / (2 * len(language_pairs))] * (
                2 * len(language_pairs)
            ) + [0.5 / (2 * len(remote_language_pairs))] * (
                2 * len(remote_language_pairs)
            )

        train_datasets = ds.interleave_datasets(
            train_datasets,
            split="train",
            seed=42,
            stopping_strategy="all_exhausted",
            probabilities=probabilities,
        )

        train_datasets = train_datasets.shuffle(seed=42, buffer_size=10_000)

        train_dataloader = DataLoader(
            train_datasets,
            collate_fn=KgMtBatch.collate_fn,
            batch_size=train_batch_size,
            num_workers=0,
        )

        val_datasets, data_info = load_local_datasets(
            val_data_paths,
            language_pairs,
            LANGUAGE_MAP[translation_model_name],
            kg_path,
            gold_probability=0.0,
        )

        val_dataloaders = [
            DataLoader(
                dataset,
                collate_fn=KgMtBatch.collate_fn,
                batch_size=val_batch_size,
                num_workers=0,
            )
            for dataset in val_datasets
        ]

        wandb_logger = WandbLogger(
            name=f"kgmt-{translation_model_name}",
            project="kgmt",
            save_dir="./logs",
        )

    elif args.test:
        test_data_paths: List[str] = args.test_data_paths
        language_pairs: List[str] = args.language_pairs
        test_batch_size: int = args.test_batch_size
        assert test_data_paths, "--test_data_paths must be specified if --test is set"
        assert language_pairs, "--language_pairs must be specified if --test is set"

        test_datasets, data_info = load_local_datasets(
            test_data_paths,
            language_pairs,
            LANGUAGE_MAP[translation_model_name],
            kg_path,
            gold_probability=0.0,
        )

        test_dataloaders = [
            DataLoader(
                dataset,
                collate_fn=KgMtBatch.collate_fn,
                batch_size=test_batch_size,
                num_workers=0,
            )
            for dataset in test_datasets
        ]

    elif args.predict:
        pred_input_paths: List[str] = args.pred_input_paths
        pred_output_paths: List[str] = args.pred_output_paths
        language_pairs: List[str] = args.language_pairs
        pred_batch_size: int = args.pred_batch_size
        assert pred_input_paths, "--pred_input_paths must be non-empty for --predict"
        assert pred_output_paths, "--pred_output_paths must be non-empty for --predict"
        assert language_pairs, "--language_pairs must be specified for --predict"
        assert len(pred_input_paths) == len(
            pred_output_paths
        ), "Number of input paths should be equal to number of output paths"
        assert len(pred_input_paths) == len(
            language_pairs
        ), "Number of input paths should be equal to number of language pairs"
        retrieval_mode = "skip" if args.disable_retriever else "gold"
        gold_probability = 1.0 if args.use_gold_entities else 0.0

        pred_datasets, data_info = load_local_datasets(
            pred_input_paths,
            language_pairs,
            LANGUAGE_MAP[translation_model_name],
            kg_path,
            mode=retrieval_mode,
            gold_probability=gold_probability,
        )

        pred_dataloaders = [
            DataLoader(
                dataset,
                collate_fn=KgMtBatch.collate_fn,
                batch_size=pred_batch_size,
                num_workers=0,
            )
            for dataset in pred_datasets
        ]

    else:
        raise Exception("No mode selected: set either --train, --test, or --predict.")

    trainer = pl.Trainer(
        default_root_dir="./logs",
        devices=args.devices,
        limit_val_batches=limit_val_batches,
        precision="16-mixed",
        accumulate_grad_batches=accumulate_grad_batches,
        gradient_clip_val=gradient_clip_val,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=None,
        max_steps=max_steps,
        max_epochs=max_epochs,
        logger=wandb_logger,
        callbacks=[
            RichProgressBar(),
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=patience,
            ),
            ModelCheckpoint(
                filename="kgmt-{step:02d}-{val_bleu:0.2f}-{val_loss:.3f}",
                save_top_k=1,
                monitor="val_loss",
                mode="min",
                every_n_train_steps=int(val_check_interval / accumulate_grad_batches),
            ),
            LearningRateMonitor(),
        ],
    )

    if checkpoint_path:
        model = KgMt.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            translation_model_name=translation_model_name,
            log_info=data_info,
            num_beams=num_beams,
        )
    else:
        model = KgMt(
            translation_model_name=translation_model_name,
            log_info=data_info,
            num_beams=num_beams,
        )

    if args.train:
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloaders,
        )

    elif args.test:
        trainer.test(model=model, dataloaders=test_dataloaders)

    elif args.predict:
        predictions = {}
        batch_predictions = trainer.predict(model=model, dataloaders=pred_dataloaders)

        for dataloader_idx, batch in batch_predictions:
            if dataloader_idx not in predictions:
                predictions[dataloader_idx] = {
                    "instance_ids": [],
                    "predictions": [],
                    "relevant_entities": [],
                    "relevant_source_names": [],
                    "relevant_target_names": [],
                }
            predictions[dataloader_idx]["instance_ids"].extend(batch.instance_ids)
            predictions[dataloader_idx]["predictions"].extend(batch.predictions)
            predictions[dataloader_idx]["relevant_entities"].extend(
                batch.relevant_entities
            )
            predictions[dataloader_idx]["relevant_source_names"].extend(
                batch.relevant_source_names
            )
            predictions[dataloader_idx]["relevant_target_names"].extend(
                batch.relevant_target_names
            )

        for dataloader_idx, pred_output_path in enumerate(pred_output_paths):
            _instance_ids = predictions[dataloader_idx]["instance_ids"]
            _predictions = predictions[dataloader_idx]["predictions"]
            _entities = predictions[dataloader_idx]["relevant_entities"]
            _source_names = predictions[dataloader_idx]["relevant_source_names"]
            _target_names = predictions[dataloader_idx]["relevant_target_names"]

            with open(pred_output_path, "w") as f_out:
                for (
                    instance_id,
                    prediction,
                    entities,
                    source_names,
                    target_names,
                ) in zip(
                    _instance_ids, _predictions, _entities, _source_names, _target_names
                ):
                    if ".json" in pred_output_path:
                        output_data = {
                            "id": instance_id,
                            "prediction": prediction,
                            "relevant_entities": entities,
                            "source_names": source_names,
                            "target_names": target_names,
                        }
                        output_data_str = json.dumps(output_data, ensure_ascii=False)
                    else:
                        output_data_str = prediction
                    f_out.write(f"{output_data_str}\n")

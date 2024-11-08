#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""
Code for the Contriever model.

Derived from `https://github.com/facebookresearch/contriever/blob/main/src/contriever.py` under CC-BY-NC 4.0 license.

Original authors of the code above are:
 - Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, Edouard Grave

The code has been modified to fit the needs of the project.
"""

import torch
from transformers import BertModel


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):
        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb


if __name__ == "__main__":
    from transformers import AutoTokenizer

    contriever = Contriever.from_pretrained("facebook/mcontriever").eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/mcontriever")

    sentences = [
        "Where was Marie Curie born?",
        "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
        "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace.",
    ]

    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    embeddings = contriever(**inputs)

    score01 = embeddings[0] @ embeddings[1]  # 1.0473
    score02 = embeddings[0] @ embeddings[2]  # 1.0095

    import torch.nn.functional as F

    print(F.cosine_similarity(embeddings[0], embeddings[1], dim=0))
    print(F.cosine_similarity(embeddings[0], embeddings[2], dim=0))

    print(f"Score: {score01:.4f}")
    print(f"Score: {score02:.4f}")

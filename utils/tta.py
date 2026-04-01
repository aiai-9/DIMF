# utils/tta.py

import torch


def horizontal_flip(x):
    return torch.flip(x, dims=[3])


def identity(x):
    return x


def get_tta_transforms():

    return [
        identity,
        horizontal_flip,
    ]


@torch.no_grad()
def tta_predict(model, batch, transforms):

    probs = []

    for t in transforms:

        batch_aug = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        for k in ["source", "target", "hint", "hint_ori"]:
            if k in batch_aug:
                batch_aug[k] = t(batch_aug[k])

        source, target, c, labels = model.get_input(batch_aug, model.first_stage_key)

        out = model(source, target, c, labels)

        d = out[1]

        if "v/probs" in d:
            p = d["v/probs"]

        elif "probs" in d:
            p = d["probs"]

        elif "v/logits" in d:
            p = torch.sigmoid(d["v/logits"])

        elif "logits" in d:
            p = torch.sigmoid(d["logits"])

        else:
            raise RuntimeError("Model output missing probabilities")

        probs.append(p.detach().cpu())

    probs = torch.stack(probs).mean(0)

    return probs.view(-1)
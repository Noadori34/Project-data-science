"""Round-trip check of kidney.sam_prompt.mmdet_to_sam.

`sam_to_mmdet` re-implements the forward conversion from the UltraSam repo [S1]
(weights/convert_sam_bb_to_mmdet.py + weights/convert_sam_to_mmdet.py). Mapping a SAM ViT-B
state dict forward and back must give the same tensors.
Run: PYTHONPATH=src python -m pytest tests/test_ultrasam_mapping.py -q
"""
import torch
from segment_anything import sam_model_registry

from kidney.sam_prompt import mmdet_to_sam

ATTN = ["layers.0.self_attn", "layers.0.cross_attn_token_to_image", "layers.0.cross_attn_image_to_token",
        "layers.1.self_attn", "layers.1.cross_attn_token_to_image", "layers.1.cross_attn_image_to_token",
        "final_attn_token_to_image"]


def sam_to_mmdet(sd):
    out = {}
    bb = {k.replace("image_encoder.", ""): v for k, v in sd.items() if "image_encoder" in k}
    bb = {k.replace("blocks.", "layers."): v for k, v in bb.items()}
    bb = {k.replace("patch_embed.proj", "patch_embed.projection"): v for k, v in bb.items()}
    bb = {k.replace("norm", "ln"): v for k, v in bb.items()}
    bb = {k.replace("neck", "channel_reduction"): v for k, v in bb.items()}
    bb = {k.replace("mlp.lin1", "ffn.layers.0.0"): v for k, v in bb.items()}
    bb = {k.replace("mlp.lin2", "ffn.layers.1"): v for k, v in bb.items()}
    out.update({f"backbone.{k}": v for k, v in bb.items()})
    md = {k: v for k, v in sd.items() if "mask_decoder." in k}
    dec = {k.replace("mask_decoder.transformer.", "decoder."): v for k, v in md.items() if "mask_decoder.transformer" in k}
    head = {k.replace("mask_decoder.", "bbox_head."): v for k, v in md.items() if "mask_decoder.transformer" not in k}
    dec = {k.replace("mlp.lin1", "mlp.layers.0.0").replace("mlp.lin2", "mlp.layers.1").replace("norm_final_attn", "post_norm"): v for k, v in dec.items()}
    dec = {k.replace("cross_attn_image_to_token", "cross_attn_image_to_token.attn").replace(
        "cross_attn_token_to_image", "cross_attn_token_to_image.attn").replace(
        "final_attn_token_to_image", "final_attn_token_to_image.attn").replace(
        "self_attn", "self_attn.attn").replace("proj.", "proj_").replace("out_proj_", "out_proj."): v for k, v in dec.items()}
    for l in ATTN:
        dec[f"decoder.{l}.attn.in_proj_bias"] = torch.cat([dec.pop(f"decoder.{l}.attn.{p}_proj_bias") for p in "qkv"])
    out.update(dec)
    pe = {k: v for k, v in sd.items() if "prompt_encoder." in k}
    pe["prompt_encoder.label_encoder.label_embedding.weight"] = torch.cat([
        pe.pop("prompt_encoder.no_mask_embed.weight"),
        *[pe.pop(f"prompt_encoder.point_embeddings.{i}.weight") for i in range(4)],
        pe.pop("prompt_encoder.not_a_point_embed.weight"),
        head.pop("bbox_head.mask_tokens.weight"), head.pop("bbox_head.iou_token.weight")])
    out.update(pe)
    out.update(head)
    return out


def test_round_trip():
    sd = sam_model_registry["vit_b"](checkpoint=None).state_dict()
    back = mmdet_to_sam(sam_to_mmdet(sd))
    assert set(back) == set(sd), (set(sd) - set(back), set(back) - set(sd))
    for k in sd:
        assert torch.equal(back[k], sd[k]), k

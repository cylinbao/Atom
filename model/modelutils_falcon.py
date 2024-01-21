from tqdm import tqdm
import torch
from transformers.models.falcon.modeling_falcon import FalconDecoderLayer
from qFalconLayer import QFalconDecoderLayer
from functools import partial
from quant import quantize_activation_wrapper, quantize_attn_v_wrapper, quantize_attn_k_wrapper

def reorder_model_falcon(model, device, args, reorder_index):
    model.config.use_cache = False
    layers = model.model.decoder.layers
    assert reorder_index is not None, "Reorder index is None"

    for i in tqdm(range(len(layers))):
        layers[i] = layers[i].to(device)
        layers[i] = layers[i].to(device)
        if isinstance(layers[i], FalconDecoderLayer):
            m = QFalconDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QFalconDecoderLayer):
            m = layers[i]
        breakpoint()
        
        nameTemplate_fc = 'decoder.layers.{}.{}.{}' # Something like layers.10.fc1
        nameTemplate_attn = 'decoder.layers.{}.{}.{}.{}' # Something like layers.10.self_attn.q_proj

        m.fc1.reorder(
            in_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc1', 'input')],
            out_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc2', 'input')]
        )
        m.fc2.reorder(
            in_reorder_index=reorder_index[nameTemplate_fc.format(i, 'fc2', 'input')],
            out_reorder_index=None
        )

        # K has outlier should be kept.
        # Output Not reorder due to the RoPE embedding.
        m.self_attn.q_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'q_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.k_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'k_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.v_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'v_proj', 'input')],
            out_reorder_index=None
        )
        m.self_attn.out_proj.reorder(
            in_reorder_index=reorder_index[nameTemplate_attn.format(i, 'self_attn', 'out_proj', 'input')],
            out_reorder_index=None
        )

        m.self_attn_layer_norm.register_buffer('reorder_index', 
            reorder_index[nameTemplate_attn.format(i, 'self_attn', 'k_proj', 'input')] # Random choose one from k,q,v proj.
        )
        if m.do_layer_norm_before:
            m.final_layer_norm.register_buffer('reorder_index',
                reorder_index[nameTemplate_fc.format(i, 'fc1', 'input')]
            )
        m.self_attn.register_buffer(
            'out_reorder_index', 
            reorder_index[nameTemplate_attn.format(i, 'self_attn', 'out_proj', 'input')]
        )

        layers[i] = layers[i].cpu()
        layers[i] = m.cpu()
        del m
        torch.cuda.empty_cache()
    return model

def add_act_quant_wrapper_falcon(model, device, args, scales):
    model.config.use_cache = False
    layers = model.transformer.h

    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], FalconDecoderLayer):
            m = QFalconDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QFalconDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)

        m.self_attention.act_quant = partial(quantize_activation_wrapper, args=args)
        m.self_attention.v_quant = partial(quantize_attn_v_wrapper, args=args)
        m.self_attention.k_quant = partial(quantize_attn_k_wrapper, args=args)

        m.mlp.act_quant = partial(quantize_activation_wrapper, args=args)
        if hasattr(m, 'ln_attn'):
            m.ln_attn.act_quant = partial(quantize_activation_wrapper, args=args)
        if hasattr(m, 'ln_mlp'):
            m.ln_mlp.act_quant = partial(quantize_activation_wrapper, args=args)
        if hasattr(m, 'input_layernorm'):
            m.input_layernorm.act_quant = partial(quantize_activation_wrapper, args=args)
        if hasattr(m, 'post_attention_layernorm'):
            m.post_attention_layernorm.act_quant = partial(quantize_activation_wrapper, args=args)
        
        layers[i] = m.cpu()
        torch.cuda.empty_cache()

    return model

def quantize_model_falcon(model, device, args):
    model.config.use_cache = False
    layers = model.transformer.h
    for i in tqdm(range(len(layers))):
        m = None
        if isinstance(layers[i], FalconDecoderLayer):
            m = QFalconDecoderLayer(
                originalLayer=layers[i],
                args=args,
            )
        elif isinstance(layers[i], QFalconDecoderLayer):
            m = layers[i]

        if m is None:
            continue

        m = m.to(device)
        m.mlp.dense_h_to_4h.quant()
        m.mlp.dense_4h_to_h.quant()
        m.self_attention.query_key_value.quant()
        m.self_attention.dense.quant()

        layers[i] = m.cpu()
        torch.cuda.empty_cache()
    return model
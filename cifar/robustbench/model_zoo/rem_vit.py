import torch
import logging
import math

import torch.nn.functional as F
from .vision_transformer_rem import VisionTransformer as VisionTransformer_REM
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import is_model, is_model_in_modules
from timm.models.helpers import load_checkpoint
from timm.models.layers import set_layer_config
from timm.models.hub import load_model_config_from_hf
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed

_logger = logging.getLogger(__name__)

IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def split_model_name(model_name):
    model_split = model_name.split(':', 1)
    if len(model_split) == 1:
        return '', model_split[0]
    else:
        source_name, model_name = model_split
        assert source_name in ('timm', 'hf_hub')
        return source_name, model_name


def safe_model_name(model_name, remove_source=True):
    def make_safe(name):
        return ''.join(c if c.isalnum() else '_' for c in name).rstrip('_')

    if remove_source:
        model_name = split_model_name(model_name)[-1]
    return make_safe(model_name)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {'vit_base_patch16_384': _cfg(
    url='https://storage.googleapis.com/vit_models/augreg/'
        'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
    # 'B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz', #copy from pytorch-image-models
    input_size=(3, 384, 384), crop_pct=1.0)
}


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None
    model = build_model_with_cfg(
        our_vit, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def _create_vision_transformer_rem(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None
    model = build_model_with_cfg(
        rem_vit, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def vit_base_patch16_384(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


def vit_base_patch16_384_rem(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer_rem('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


def create_model_rem(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name == 'hf_hub':
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = vit_base_patch16_384_rem(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model


def create_model(
        model_name,
        pretrained=False,
        checkpoint_path='',
        scriptable=None,
        exportable=None,
        no_jit=None,
        **kwargs):
    """Create a model

    Args:
        model_name (str): name of model to instantiate
        pretrained (bool): load pretrained ImageNet-1k weights if true
        checkpoint_path (str): path of checkpoint to load after model is initialized
        scriptable (bool): set layer config so that model is jit scriptable (not working for all models yet)
        exportable (bool): set layer config so that model is traceable / ONNX exportable (not fully impl/obeyed yet)
        no_jit (bool): set layer config so that model doesn't utilize jit scripted layers (so far activations only)

    Keyword Args:
        drop_rate (float): dropout rate for training (default: 0.0)
        global_pool (str): global pool type (default: 'avg')
        **: other kwargs are model specific
    """
    source_name, model_name = split_model_name(model_name)

    # Only EfficientNet and MobileNetV3 models have support for batchnorm params or drop_connect_rate passed as args
    is_efficientnet = is_model_in_modules(model_name, ['efficientnet', 'mobilenetv3'])
    if not is_efficientnet:
        kwargs.pop('bn_tf', None)
        kwargs.pop('bn_momentum', None)
        kwargs.pop('bn_eps', None)

    # handle backwards compat with drop_connect -> drop_path change
    drop_connect_rate = kwargs.pop('drop_connect_rate', None)
    if drop_connect_rate is not None and kwargs.get('drop_path_rate', None) is None:
        print("WARNING: 'drop_connect' as an argument is deprecated, please use 'drop_path'."
              " Setting drop_path to %f." % drop_connect_rate)
        kwargs['drop_path_rate'] = drop_connect_rate

    # Parameters that aren't supported by all models or are intended to only override model defaults if set
    # should default to None in command line args/cfg. Remove them if they are present and not set so that
    # non-supporting models don't break and default args remain in effect.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    if source_name == 'hf_hub':
        # For model names specified in the form `hf_hub:path/architecture_name#revision`,
        # load model weights + default_cfg from Hugging Face hub.
        hf_default_cfg, model_name = load_model_config_from_hf(model_name)
        kwargs['external_default_cfg'] = hf_default_cfg  # FIXME revamp default_cfg interface someday

    with set_layer_config(scriptable=scriptable, exportable=exportable, no_jit=no_jit):
        model = vit_base_patch16_384(pretrained=pretrained, **kwargs)

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)

    return model


class our_vit(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x, mask_token=None, unmask=None, return_norm=False):
        x = self.patch_embed(x)
        B, N, _ = x.shape
        device = x.device
        # switch input tokens by mask_token
        if mask_token is not None:
            mask_tokens = mask_token.module.expand(B, N, -1).to(device)
            unmask = unmask.unsqueeze(2).to(device)  # mask_chosed
            x = x * (1 - unmask) + mask_tokens * unmask
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x = self.blocks(x)
        x = self.norm(x)

        x_norm = x
        if self.dist_token is None and return_norm is False:
            return self.pre_logits(x[:, 0])
        elif self.dist_token is None and return_norm is not False:
            return self.pre_logits(x[:, 0]), x_norm
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, mask_token=None, unmask=None, return_norm=False):
        if return_norm:
            x, x_norm = self.forward_features(x, mask_token, unmask, return_norm)
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x, x_norm
        else:
            x = self.forward_features(x)
            if self.head_dist is not None:
                x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
                if self.training and not torch.jit.is_scripting():
                    return x, x_dist
                else:
                    return (x + x_dist) / 2
            else:
                x = self.head(x)
            return x


class rem_vit(VisionTransformer_REM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward_features(self, x, len_keep=None, return_attn=False):
        x = self.patch_embed(x)
        B, N, _ = x.shape
        device = x.device
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        if len_keep is not None:
            B, _, D = x.shape  # batch, length, dim
            cls_save = x[:, 0, :].unsqueeze(dim=1)
            x = x[:, 1:, :]
            x = torch.gather(x, dim=1, index=len_keep.unsqueeze(-1).repeat(1, 1, D))
            x = torch.cat((cls_save, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                x, attn = blk(x, return_attn=True)

        x = self.norm(x)
        if return_attn:
            return x[:,0], attn
        else:
            return x[:,0]

    def forward(self, x, len_keep=None, return_attn=False):
        if return_attn is True:
            x, attn = self.forward_features(x, len_keep, True)
            x = self.head(x)
            return x, attn
        else:
            x = self.forward_features(x, len_keep, False)
            x = self.head(x)
            return x


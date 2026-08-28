from types import SimpleNamespace

from backend.FarmFederate_Colab_Complete import get_hidden_dim


def test_distilbert_uses_encoder_width_not_ffn_width():
    config = SimpleNamespace(dim=768, hidden_dim=3072)

    assert get_hidden_dim(config) == 768


def test_swin_uses_final_stage_width_not_patch_embedding_width():
    config = SimpleNamespace(
        embed_dim=96,
        hidden_sizes=[96, 192, 384, 768],
    )

    assert get_hidden_dim(config) == 768

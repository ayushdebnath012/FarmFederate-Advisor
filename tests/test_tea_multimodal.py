import hashlib

import torch

import tea_train as training


def test_tokenizer_uses_stable_hash_ids():
    tokenizer = training.SimpleTokenizer()
    word = "blight"
    digest = hashlib.blake2b(
        word.encode("utf-8"), digest_size=8, person=b"FarmFed"
    ).digest()
    expected = (
        int.from_bytes(digest, byteorder="big", signed=False)
        % (tokenizer.vocab_size - 103)
    ) + 103
    assert tokenizer.tokenize(word) == [
        tokenizer.cls_token_id,
        expected,
        tokenizer.sep_token_id,
    ]


def test_grouped_split_has_no_source_overlap():
    labels = [0, 0, 1, 1, 0, 1, 2, 2, 2, 0, 1, 2]
    groups = ["a", "a", "b", "b", "c", "d", "e", "e", "f", "g", "h", "i"]
    train_idx, val_idx = training.grouped_stratified_split(
        labels, groups, val_split=0.34, seed=7
    )
    assert set(train_idx).isdisjoint(val_idx)
    assert set(groups[i] for i in train_idx).isdisjoint(
        groups[i] for i in val_idx
    )
    assert sorted(train_idx + val_idx) == list(range(len(labels)))


def test_grouped_three_way_split_has_no_source_overlap():
    labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2] * 3
    groups = [f"source_{i // 2}" for i in range(len(labels))]
    train_idx, val_idx, test_idx = training.grouped_train_val_test_split(
        labels,
        groups,
        val_split=0.20,
        test_split=0.20,
        seed=11,
    )
    partitions = [
        {groups[i] for i in indices}
        for indices in (train_idx, val_idx, test_idx)
    ]
    assert partitions[0].isdisjoint(partitions[1])
    assert partitions[0].isdisjoint(partitions[2])
    assert partitions[1].isdisjoint(partitions[2])
    assert sorted(train_idx + val_idx + test_idx) == list(range(len(labels)))


def test_target_shortcut_sanitizer_is_fit_on_training_only():
    frame = training.pd.DataFrame(
        {
            "text": [
                "grayword shared symptom",
                "grayword shared lesion",
                "grayword shared patch",
                "redword shared symptom",
                "redword shared lesion",
                "redword shared patch",
            ],
            "labels": [[0], [0], [0], [4], [4], [4]],
        }
    )
    blocked = training.fit_label_leakage_vocabulary(
        frame, min_count=3, purity_threshold=0.95
    )
    cleaned = training.sanitize_annotation_text(frame, blocked)
    assert "grayword" in blocked
    assert "redword" in blocked
    assert "shared" not in blocked
    assert all("grayword" not in text for text in cleaned["text"])
    assert all("redword" not in text for text in cleaned["text"])


def test_fedavg_preserves_integer_buffers():
    global_state = {
        "weight": torch.tensor([0.0]),
        "num_batches_tracked": torch.tensor(0, dtype=torch.long),
    }
    clients = [
        {
            "weight": torch.tensor([1.0]),
            "num_batches_tracked": torch.tensor(2, dtype=torch.long),
        },
        {
            "weight": torch.tensor([3.0]),
            "num_batches_tracked": torch.tensor(5, dtype=torch.long),
        },
    ]
    state = training.fedavg_aggregate(global_state, clients, [1, 3])
    assert torch.allclose(state["weight"], torch.tensor([2.5]))
    assert state["num_batches_tracked"].dtype == torch.long
    assert state["num_batches_tracked"].item() == 5


def test_multimodal_forward_supports_ablation_and_alignment_loss():
    model = training.MultiModalClassifier(
        num_labels=training.NUM_CLASSES,
        max_seq_len=8,
        modality_dropout=0.0,
        image_only_probability=0.0,
        text_only_probability=0.0,
        vision_backbone="lightweight",
        pretrained_vision=False,
    )
    model.eval()
    input_ids = torch.tensor(
        [[101, 500, 102, 0, 0, 0, 0, 0], [101, 700, 102, 0, 0, 0, 0, 0]]
    )
    attention_mask = (input_ids != 0).long()
    pixels = torch.randn(2, 3, 64, 64)
    labels = torch.eye(training.NUM_CLASSES)[:2]

    both = model(input_ids, attention_mask, pixels, labels=labels)
    text_only = model(
        input_ids,
        attention_mask,
        pixels,
        modality_mask=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
    )
    image_only_labeled = model(
        input_ids,
        attention_mask,
        pixels,
        labels=labels,
        modality_mask=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
    )

    assert both["logits"].shape == (2, training.NUM_CLASSES)
    assert torch.isfinite(both["loss"])
    assert torch.isfinite(both["loss_components"]["alignment"])
    assert torch.allclose(
        text_only["modality_weights"][:, 0], torch.ones(2), atol=1e-6
    )
    assert image_only_labeled["loss_components"]["text_auxiliary"].item() == 0.0
    assert image_only_labeled["loss_components"]["alignment"].item() == 0.0
    assert torch.isfinite(
        image_only_labeled["loss_components"]["vision_auxiliary"]
    )


def test_cached_dataset_restores_fp32_spatial_features():
    dataset = training.CachedMultiModalDataset(
        input_ids=torch.ones(2, 4, dtype=torch.long),
        attention_mask=torch.ones(2, 4, dtype=torch.long),
        labels_tensor=torch.eye(training.NUM_CLASSES)[:2],
        vision_feature_maps=torch.ones(2, 2048, 2, 2, dtype=torch.float16),
        primary_labels=[0, 1],
        pairing_coverage=1.0,
    )
    item = dataset[0]
    assert item["vision_feature_map"].dtype == torch.float32
    assert dataset.labels == [0, 1]
    assert dataset.pairing_coverage == 1.0

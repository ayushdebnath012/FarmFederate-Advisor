# ============================================================================
# STEP 14: TRAIN FUSION MODEL (Text + Image Combined)
# ============================================================================
print("#"*70)
print("TRAINING MULTIMODAL FUSION MODEL (Text + Image Combined)")
print("#"*70)

# Fusion model configurations to test
FUSION_CONFIGS = [
    {'name': 'Fusion-Concat', 'fusion_type': 'concat', 'text_model': 'bert-base-uncased', 'vit_model': 'google/vit-base-patch16-224'},
    {'name': 'Fusion-Gated', 'fusion_type': 'gated', 'text_model': 'roberta-base', 'vit_model': 'google/vit-base-patch16-224'},
]

# Store fusion results
fusion_results = {
    'federated': {},
    'centralized': {},
}

# Prepare matched text-image pairs for fusion
min_fusion_samples = min(len(text_data), len(image_data))
fusion_texts = text_data[:min_fusion_samples]
fusion_images = image_data[:min_fusion_samples]
fusion_labels = text_labels[:min_fusion_samples]

print(f"Fusion dataset size: {min_fusion_samples} matched text-image pairs")

for config in FUSION_CONFIGS:
    model_name = config['name']
    print(f"\n{'='*60}\nFusion Model: {model_name}\n{'='*60}")

    try:
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config['text_model'])
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create enhanced dataset with sensor fusion
        n_train = int(0.8 * min_fusion_samples)

        # Validation set
        val_dataset = EnhancedMultiModalDataset(
            texts=fusion_texts[n_train:n_train+300],
            images=fusion_images[n_train:n_train+300],
            labels=fusion_labels[n_train:n_train+300],
            tokenizer=tokenizer,
            image_transform=image_transform,
            use_sensor_fusion=True,
            use_weak_labels=True
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

        # =====================================================================
        # FEDERATED TRAINING WITH FUSION MODEL
        # =====================================================================
        print("\n[FEDERATED FUSION]")

        # Initialize global fusion model
        fed_fusion = MultiModalFusionModel(
            text_model_name=config['text_model'],
            vit_model_name=config['vit_model'],
            num_labels=NUM_LABELS,
            fusion_type=config['fusion_type'],
            use_lora=True,
            freeze_text=True,
            freeze_vision=True
        ).to(DEVICE)

        # Initialize EMA
        ema = EMAModel(fed_fusion, decay=0.997)

        # Create client datasets
        chunk_size = n_train // NUM_CLIENTS
        client_datasets = []
        for i in range(NUM_CLIENTS):
            start = i * chunk_size
            end = start + chunk_size
            ds = EnhancedMultiModalDataset(
                texts=fusion_texts[start:end],
                images=fusion_images[start:end],
                labels=fusion_labels[start:end],
                tokenizer=tokenizer,
                image_transform=image_transform,
                use_sensor_fusion=True,
                use_weak_labels=True
            )
            client_datasets.append(ds)

        fed_history = []
        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.02)

        for rnd in range(FED_ROUNDS):
            client_states = []
            client_sizes = []

            for cid, cds in enumerate(client_datasets):
                # Clone global model for client
                client_model = MultiModalFusionModel(
                    text_model_name=config['text_model'],
                    vit_model_name=config['vit_model'],
                    num_labels=NUM_LABELS,
                    fusion_type=config['fusion_type'],
                    use_lora=True
                ).to(DEVICE)

                # Load global weights
                client_model.load_state_dict(fed_fusion.state_dict())

                # Train locally
                client_loader = DataLoader(cds, batch_size=4, shuffle=True)
                optimizer = torch.optim.AdamW(
                    [p for p in client_model.parameters() if p.requires_grad],
                    lr=2e-5
                )

                client_model.train()
                for _ in range(LOCAL_EPOCHS):
                    for batch in client_loader:
                        input_ids = batch['input_ids'].to(DEVICE)
                        attention_mask = batch['attention_mask'].to(DEVICE)
                        pixel_values = batch['pixel_values'].to(DEVICE)
                        labels = batch['labels'].to(DEVICE)
                        sensor_priors = batch['sensor_priors'].to(DEVICE)

                        logits = client_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values,
                            sensor_priors=sensor_priors
                        )

                        loss = criterion(logits, labels)
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                # Save client state
                client_states.append({k: v.cpu().clone() for k, v in client_model.state_dict().items()})
                client_sizes.append(len(cds))

                del client_model
                torch.cuda.empty_cache()

            # Apply client dropout
            keep_indices = [i for i in range(len(client_states)) if np.random.random() > 0.1]
            if len(keep_indices) == 0:
                keep_indices = [0]

            kept_states = [client_states[i] for i in keep_indices]
            kept_sizes = [client_sizes[i] for i in keep_indices]

            # FedAvg aggregation
            total_size = sum(kept_sizes)
            avg_state = {}
            for key in kept_states[0].keys():
                avg_state[key] = sum(
                    kept_states[i][key].float() * (kept_sizes[i] / total_size)
                    for i in range(len(kept_states))
                )

            fed_fusion.load_state_dict(avg_state)

            # Update EMA
            ema.update(fed_fusion)

            # Evaluate
            fed_fusion.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    pixel_values = batch['pixel_values'].to(DEVICE)
                    labels = batch['labels']
                    sensor_priors = batch['sensor_priors'].to(DEVICE)

                    logits = fed_fusion(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        sensor_priors=sensor_priors
                    )
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            preds_binary = (all_preds > 0.5).astype(int)

            metrics = {
                'f1_macro': f1_score(all_labels, preds_binary, average='macro', zero_division=0),
                'accuracy': accuracy_score(all_labels, preds_binary),
                'precision': precision_score(all_labels, preds_binary, average='macro', zero_division=0),
                'recall': recall_score(all_labels, preds_binary, average='macro', zero_division=0)
            }
            fed_history.append(metrics)
            print(f"  Round {rnd+1}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

        fusion_results['federated'][model_name] = {
            'history': fed_history,
            'final': fed_history[-1],
            'config': config
        }

        del fed_fusion, ema
        gc.collect()
        torch.cuda.empty_cache()

        # =====================================================================
        # CENTRALIZED TRAINING WITH FUSION MODEL
        # =====================================================================
        print("\n[CENTRALIZED FUSION]")

        cent_fusion = MultiModalFusionModel(
            text_model_name=config['text_model'],
            vit_model_name=config['vit_model'],
            num_labels=NUM_LABELS,
            fusion_type=config['fusion_type'],
            use_lora=True
        ).to(DEVICE)

        # Full training dataset
        full_dataset = EnhancedMultiModalDataset(
            texts=fusion_texts[:n_train],
            images=fusion_images[:n_train],
            labels=fusion_labels[:n_train],
            tokenizer=tokenizer,
            image_transform=image_transform,
            use_sensor_fusion=True,
            use_weak_labels=True
        )
        train_loader = DataLoader(full_dataset, batch_size=8, shuffle=True)

        optimizer = torch.optim.AdamW(
            [p for p in cent_fusion.parameters() if p.requires_grad],
            lr=3e-5
        )
        criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.02)

        cent_history = []
        for epoch in range(CENT_EPOCHS):
            cent_fusion.train()
            for batch in train_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                pixel_values = batch['pixel_values'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                sensor_priors = batch['sensor_priors'].to(DEVICE)

                logits = cent_fusion(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    sensor_priors=sensor_priors
                )

                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Evaluate
            cent_fusion.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(DEVICE)
                    attention_mask = batch['attention_mask'].to(DEVICE)
                    pixel_values = batch['pixel_values'].to(DEVICE)
                    labels = batch['labels']
                    sensor_priors = batch['sensor_priors'].to(DEVICE)

                    logits = cent_fusion(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        sensor_priors=sensor_priors
                    )
                    preds = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.numpy())

            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            preds_binary = (all_preds > 0.5).astype(int)

            metrics = {
                'f1_macro': f1_score(all_labels, preds_binary, average='macro', zero_division=0),
                'accuracy': accuracy_score(all_labels, preds_binary),
                'precision': precision_score(all_labels, preds_binary, average='macro', zero_division=0),
                'recall': recall_score(all_labels, preds_binary, average='macro', zero_division=0)
            }
            cent_history.append(metrics)
            print(f"  Epoch {epoch+1}: F1={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")

        fusion_results['centralized'][model_name] = {
            'history': cent_history,
            'final': cent_history[-1],
            'config': config
        }

        # Summary
        fed_f1_fusion = fusion_results['federated'][model_name]['final']['f1_macro']
        cent_f1_fusion = fusion_results['centralized'][model_name]['final']['f1_macro']
        gap = (cent_f1_fusion - fed_f1_fusion) / cent_f1_fusion * 100 if cent_f1_fusion > 0 else 0

        print(f"\n  SUMMARY: Fed={fed_f1_fusion:.4f}, Cent={cent_f1_fusion:.4f}, Gap={gap:.1f}%")

        del cent_fusion, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*70)
print("FUSION MODEL TRAINING COMPLETE!")
print("="*70)

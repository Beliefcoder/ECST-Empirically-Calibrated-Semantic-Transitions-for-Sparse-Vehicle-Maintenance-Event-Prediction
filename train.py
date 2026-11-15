import argparse
import json
from pathlib import Path

import numpy as np
import torch

import models.model as model_module
from metrics import evaluate
from utils import Dataset, load_priors, load_sparse, read_config, seed_everything


def predict(model, data):
    model.eval()
    outputs = []
    with torch.no_grad():
        for visits, lengths, _ in data:
            outputs.append(model(visits, lengths).sigmoid().cpu().numpy())
    return np.vstack(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/salt_qwen25.json')
    parser.add_argument('--data-root', default='data/qwen25')
    parser.add_argument('--output', default='outputs/salt_seed50')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--mode', choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config['seed'])
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    model_module.device = device
    priors = load_priors(args.data_root)
    model = model_module.Model(priors[0].shape[0], *priors, dropout=config['dropout']).to(device)
    standard = Path(args.data_root) / 'standard'
    frequency = load_sparse(standard / 'train/code_y.npz').sum(0)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    if args.mode == 'train':
        train = Dataset(standard / 'train', config['batch_size'], device, shuffle=True)
        valid = Dataset(standard / 'valid', config['batch_size'], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        best = -1.0
        history = []
        for epoch in range(1, config['epochs'] + 1):
            model.train()
            for visits, lengths, labels in train:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(model(visits, lengths), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
                optimizer.step()
            train.on_epoch_end()
            metrics = evaluate(valid.labels, predict(model, valid), frequency)
            history.append({'epoch': epoch, **metrics})
            print(history[-1], flush=True)
            if metrics['W-F1'] > best:
                best = metrics['W-F1']
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_f1': best}, output / 'best_checkpoint.pt')
            (output / 'validation.json').write_text(json.dumps(history, indent=2))
        checkpoint = output / 'best_checkpoint.pt'
    else:
        if not args.checkpoint:
            parser.error('--checkpoint is required in eval mode')
        checkpoint = args.checkpoint
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['model_state_dict'], strict=True)
    test = Dataset(standard / 'test', config['batch_size'], device)
    scores = predict(model, test)
    result = {'epoch': state['epoch'], **evaluate(test.labels, scores, frequency)}
    np.savez_compressed(output / 'test_predictions.npz', scores=scores, labels=test.labels)
    (output / 'test_metrics.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    main()
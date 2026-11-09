"""Prompt-level split isolation and safe replacement of existing extractions."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from dalg.llm.prompt_extraction import run_prompt_extraction

spec = importlib.util.spec_from_file_location(
    'sae_prompt_holdout', Path(__file__).resolve().parents[1] / 'scripts/temporary/reextract_sae_prompt_holdout.py',
)
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)


def records(subset, count=20, offset=0):
    return [
        {'row_id': offset + i, 'subset': subset, 'prompt': f'{subset} {i // 2} template {i % 2}',
         'labels': {'value': i // 2, 'template': i % 2}}
        for i in range(count)
    ]


def test_prompt_split_is_disjoint_exhaustive_and_consistent_across_variants():
    temperature = records('temperature')
    years = records('years', offset=len(temperature))
    combined = temperature + years
    development, test = holdout.prompt_split(combined)
    assert len(development) == 32 and len(test) == 8
    assert set(development).isdisjoint(test)
    assert sorted(development + test) == list(range(40))
    assert holdout.prompt_split(combined) == (development, test)
    for subset in [temperature, years]:
        standalone = holdout.prompt_split(subset)[1]
        assert {subset[i]['prompt'] for i in standalone} == {
            combined[i]['prompt'] for i in test if combined[i]['subset'] == subset[0]['subset']
        }
    # Row-level sampling permits a value to occur in both populations.
    dev_values = {combined[i]['labels']['value'] for i in development if combined[i]['subset'] == 'years'}
    test_values = {combined[i]['labels']['value'] for i in test if combined[i]['subset'] == 'years'}
    assert dev_values & test_values


def test_failed_test_extraction_preserves_original_then_resumes_and_replaces(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaModel, PreTrainedTokenizerFast

    torch.set_num_threads(2)
    checkpoint = tmp_path / 'tiny_llama'
    backend = Tokenizer(models.WordLevel({'<unk>': 0, '<bos>': 1, '<eos>': 2, 'years': 3, 'template': 4}, unk_token='<unk>'))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>', bos_token='<bos>', eos_token='<eos>')
    tokenizer.save_pretrained(checkpoint)
    model = LlamaModel(LlamaConfig(vocab_size=len(tokenizer), hidden_size=8, intermediate_size=16,
                                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                  max_position_embeddings=32, bos_token_id=1, eos_token_id=2))
    model.save_pretrained(checkpoint)
    root = tmp_path / 'sae-manifold'
    name = 'years_neutral'
    dataset = root / 'dataset' / name
    dataset.mkdir(parents=True)
    source = records('years')
    (dataset / 'prompts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in source))
    (dataset / 'metadata.json').write_text(json.dumps({'counts': {'years': 20}, 'total_rows': 20}))
    destination = holdout.activation_path(root, name)
    # Remote previews must resolve the cached revision even when AutoConfig omits it.
    revision = 'a' * 40
    with monkeypatch.context() as context:
        context.setattr('transformers.AutoConfig.from_pretrained', lambda *args, **kwargs: model.config)
        context.setattr('transformers.AutoTokenizer.from_pretrained', lambda *args, **kwargs: tokenizer)
        context.setattr('transformers.utils.hub.cached_file',
                        lambda *args, **kwargs: f'/cache/snapshots/{revision}/config.json')
        preview = run_prompt_extraction(dataset=dataset, out_dir=destination, model_name='test/llama',
                                        revision=revision, preview=True)
        assert preview['model_revision'] == revision
        assert not destination.exists()
    run_prompt_extraction(dataset=dataset, out_dir=destination, model_name=str(checkpoint),
                          device='cpu', dtype='float32', shard_size=8)
    fitted = destination / 'old_model' / 'model.pt'
    fitted.parent.mkdir()
    fitted.write_bytes(b'keep fitted model')
    old_tensor = destination / 'layer00/shard_00000.pt'
    original_hash = holdout._sha256(old_tensor)
    plan = holdout.prepare(root, name)
    assert holdout.prepare(root, name) == plan
    with pytest.raises(FileExistsError):
        holdout.prepare(root, name, fraction=0.3)
    holdout.run(root, name, preview=True, device='cpu')
    assert not (root / f'.reextract_{name}').exists()

    def fail_test(**kwargs):
        if Path(kwargs['dataset']).name == 'test':
            raise RuntimeError('simulated extraction failure')
        return run_prompt_extraction(**kwargs)

    monkeypatch.setattr(holdout, 'run_prompt_extraction', fail_test)
    with pytest.raises(RuntimeError, match='simulated'):
        holdout.run(root, name, device='cpu')
    assert holdout._sha256(old_tensor) == original_hash
    assert fitted.read_bytes() == b'keep fitted model'
    assert not (root / 'pre_holdout').exists()
    monkeypatch.setattr(holdout, 'run_prompt_extraction', run_prompt_extraction)
    holdout.run(root, name, device='cpu')
    assert json.loads((destination / 'config.json').read_text())['num_rows'] == 16
    assert json.loads((destination / 'test/config.json').read_text())['num_rows'] == 4
    archive = root / 'pre_holdout' / destination.name
    assert (archive / 'old_model/model.pt').read_bytes() == b'keep fitted model'
    assert not list(archive.glob('layer*'))
    assert not (destination / 'old_model').exists()
    # A repeated job validates the completed replacement without recomputing it.
    with monkeypatch.context() as context:
        context.setattr(holdout, 'run_prompt_extraction', lambda **kwargs: pytest.fail('unexpected extraction'))
        holdout.run(root, name, device='cpu')

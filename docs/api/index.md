# API Reference

The complete public API, generated from the source docstrings.
Core symbols are importable from the top-level `kamui` package; the
distributed-training helpers live in `kamui.training`.

## Configuration & Model

::: kamui.model.config.ModelConfig

::: kamui.model.transformer.KAMUITransformer

## Tokenizer

::: kamui.tokenizer.bpe.BPETokenizer

::: kamui.tokenizer.vocab.Vocabulary

## Training

::: kamui.training.trainer.TrainingConfig

::: kamui.training.trainer.Trainer

## Distributed Training (v0.3)

::: kamui.training.distributed.wrap_ddp

::: kamui.training.distributed.DistributedDataLoader

::: kamui.training.distributed.spawn_workers

::: kamui.training.distributed.all_reduce_mean

## Hooks

::: kamui.hooks.manager.HookManager

::: kamui.hooks.registry.HookRegistry

## Interpretability Tools

::: kamui.mechinterp.attention_viz.AttentionVisualizer

::: kamui.mechinterp.logit_lens.LogitLens

::: kamui.mechinterp.probing.LinearProbe

::: kamui.mechinterp.activation_patch.ActivationPatcher

::: kamui.mechinterp.induction.InductionHeadDetector

::: kamui.mechinterp.circuits.CircuitAblator

## Interpretability Tools (v0.2)

::: kamui.mechinterp.attribution.GradientAttribution

::: kamui.mechinterp.superposition.SparseAutoencoder

::: kamui.mechinterp.superposition.interpret_features

::: kamui.mechinterp.steering.FeatureSteerer

::: kamui.mechinterp.steering.build_steering_vector

## Evaluation

::: kamui.evaluate.perplexity.compute_perplexity

::: kamui.evaluate.generation.generate

::: kamui.evaluate.calibration.expected_calibration_error

"""Reproducibility utilities: seeding and determinism.

Responsibilities:
    - ``set_seed(seed: int)``:
        Set all random seeds for reproducibility:
            random.seed(seed)
            numpy.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    - ``set_deterministic()``:
        Configure PyTorch for fully deterministic operation:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

        Note: deterministic mode is slower than the default.  Use it
        for debugging and regression testing, not for production training.

    - ``get_device() -> torch.device``:
        Return the best available device: CUDA if available, else MPS
        (Apple Silicon), else CPU.  Logs which device was selected.

Why this module exists:
    Unreproducible results are a common failure mode in ML research.
    A fixed seed and deterministic ops guarantee that two runs with the
    same config produce byte-identical outputs.  This is critical for:
    - Debugging (isolate whether a change in results came from the code
      or from random variation)
    - Regression tests (verify that a refactor didn't change model output)
    - Paper reproducibility (other researchers can replicate your results)

Implemented in: Phase 1
"""

# Implementation begins in Phase 1.

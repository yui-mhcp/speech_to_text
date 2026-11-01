# :yum: CHANGELOG :yum:

This file tracks the major updates of the [speech_to_text](https://github.com/yui-mhcp/speech_to_text) project. For the global, cross-project overview of all the `yui-mhcp` repositories, see the [central CHANGELOG](https://github.com/yui-mhcp/yui-mhcp/blob/main/CHANGELOG.md) :smile:

## [v1.1.0] - 01/11/2026 - November update ! :yum:

Minor release : the TensorRT-LLM accelerated inference now relies on the `LLM API`, without any engine build step !

### Major updates

- `Whisper` supports the TensorRT-LLM `LLM API` (`runtime = 'trt_llm_api'`, `tensorrt_llm >= 1.3`) : the `transformers` checkpoint is loaded directly, the raw audio is forwarded to the engine (which computes the log-mel on GPU), and the beam search / timestamps rules are correctly handled
    - Known limitation : `condition_on_previous_text` is not supported by this runtime (the `LLM API` requires the decoder prompt to start with `<|startoftranscript|>`)
- **[BREAKING CHANGE]** The `convert_checkpoint-0.18.py` / `convert_checkpoint-0.19.py` scripts (legacy TensorRT-LLM engines) have been removed
- The installation guide has been updated for `tensorrt_llm 1.3.0rc29`, and the `numpy < 2` pin has been removed
- The shared `utils` / `loggers` / `models/core` foundation has been synced (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md) and the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) : new `image_augmentation` module, `Tokenizer` support of `transformers 5`, new `TensorRTLLMAPIRuntime` (TensorRT-LLM `LLM API`), new append-only `JSONLDatabase`, `keras 3.16` compatibility and an extended test suite

### Minor updates

- The number of mel channels is read from the checkpoint `config.json` instead of being guessed from its name (fixes `large-v1` / `large-v2` and the custom checkpoints), only the `*.en` checkpoints are considered english-only, and the official short names (e.g., `base`) resolve their tokenizer from `openai/whisper-<name>`
- The README documents the `transcribe` / `translate` functions, and the keyword search through the partial alignment of `edit_distance`
- `keras 3.16` compatibility of the shared `architectures` : the RNN masking is explicitly re-enabled (`enable_rnn_masking`), `backend_rnn` resolves the moved `keras.src.backend.rnn` module, and the masks are handled through `get_keras_mask` / `set_keras_mask` instead of the private `_keras_mask` attribute

### Bugs fixed

- `pip install -e .` works again with the recent `setuptools` versions : the `License ::` classifier, superseded by the PEP 639 `license` expression (`AGPL-3.0-or-later`), has been removed from the `pyproject.toml`

## [v1.0.0] - 21/07/2026 - Major refactoring release ! :yum:

First tagged release (`v1.0.0`) following a large refactoring of the STT stack, built on top of the shared `BaseModel` foundation (see the [base_dl_project CHANGELOG](https://github.com/yui-mhcp/base_dl_project/blob/main/CHANGELOG.md)) and the `utils` / `loggers` socle (see the [data_processing CHANGELOG](https://github.com/yui-mhcp/data_processing/blob/main/CHANGELOG.md)) !

### Major updates

- The installation now relies on `pyproject.toml` instead of `requirements.txt` : install with `pip install -e .[tf]` (or `[torch]`) and pick the optional helpers you need (`image`, `datasets`, `dev`)
- The backend is no longer hard-coded : it is selected at runtime through the `KERAS_BACKEND` environment variable, and the base install ships no backend
- The test suite has been migrated to `pytest`, mirroring the `utils` / `loggers` and model tree, with auto-skipped markers (`tensorflow`, `torch`, `keras`, `cv2`, `gpu`, `slow`, ...)
- The minimum supported Python is now `3.12`, and the version is single-sourced in `__version__`
- **[BREAKING CHANGE]** the model interfaces have been fully restructured (inherited from `base_dl_project`) : the old `models/interfaces/base_*_model.py` classes are gone. `BaseModel` now lives in `models/core/base_model.py` and is composed from small, reusable **mixins** in `models/core/mixins/` (`audio`, `image`, `text`, `classification`, `processing`, `checkpoint`, `training`)
- **[BREAKING CHANGE]** the STT models now live in `models/stt/` : `base_stt.py` (abstract `BaseSTT`) and `whisper.py` (the `Whisper` keras implementation)

### Minor updates

- `TensorRT-LLM` accelerated inference for `Whisper` has been updated to the `1.2.1` library (Python 3.12), replacing the previous `0.15.0` / Python 3.10 setup ; the `convert_checkpoint-0.18.py` / `convert_checkpoint-0.19.py` helper scripts convert HuggingFace checkpoints to TensorRT-LLM engines
- The `speech_to_text.ipynb` notebook has been reworked around the new `BaseSTT` / `Whisper` interface (build, transcription and word search)
- The `Whisper` tokenizer is now copied from the `transformers` library (the official `openai` code moved to a custom `tiktoken` tokenizer)
- `docker/` provides ready-to-use compose files (`jupyter`, `experiments`, `maggie`) and Dockerfiles for a containerized setup

# Architecture

## Goal

This project trains a Hebrew grapheme-to-phoneme (G2P) model that converts full Hebrew sentences into IPA strings.

V1 is intentionally simple:

- character-level decoder labels
- CTC training
- Hugging Face `Trainer`
- sentence-level output with spaces and punctuation preserved

## Design Principles

- Keep the code path short and explicit.
- Prefer one stable preprocessing contract over flexible-but-unclear behavior.
- Use strict preprocessing by default so invalid labels are caught early.
- Keep training compatible with standard Hugging Face tooling.

## Data Flow

1. Raw source data is stored as TSV:
   `hebrew_text<TAB>ipa_text`
2. Data preparation code under `src/` normalizes the Hebrew side and writes:
   - `dataset/train.txt`
   - `dataset/val.txt`
3. Tokenization/preprocessing code under `src/` converts TSV rows into tokenized Arrow datasets:
   - `encoder_ids`
   - `encoder_mask`
   - `decoder_ids`
4. Dataset-loading code under `src/` loads the Arrow datasets and pads batches for training.
5. Training code under `src/` trains the model with Hugging Face `Trainer`.
6. Inference code under `src/` runs greedy decoding from a saved checkpoint.

## Project Layout

- `src/`: application code for preprocessing, tokenization, modeling, training, evaluation, and inference
- `dataset/`: generated train/validation text files and tokenized caches
- `docs/`: design and operational documentation
- `plans/`: research notes, experiments, and validation plans

## Tokenization

### Decoder Side

The decoder is character-level in V1.

- Each IPA character is encoded as one token.
- Spaces are preserved as real output tokens.
- Punctuation is preserved.
- Special tokens exist for:
  - CTC blank
  - pad
  - unknown

Composite phonemes such as `ts`, `tʃ`, and `dʒ` are represented as multiple output characters in V1. This keeps the decoder logic simple.

### Encoder Side

The encoder tokenizer is loaded from the Hugging Face model repository using the raw `tokenizer.json` path, which avoids the known tokenizer issues with the Dicta character-level model.

## Model

The current model lives in the modeling layer under `src/`.

Pipeline:

1. Load the Hebrew encoder from `dicta-il/dictabert-large-char-menaked`
2. If the returned model is wrapped, unwrap the base BERT at `.bert`
3. Run the encoder on `input_ids` and `attention_mask`
4. Project encoder hidden states to a smaller dimension (`256` by default)
5. Upsample the time axis with `repeat_interleave` (`2x` by default)
6. Apply a linear classifier to the decoder vocabulary size
7. Compute CTC loss when labels are present

This produces a simple non-autoregressive model that is compatible with greedy decoding.

## Training

Training uses Hugging Face `Trainer` from the training layer under `src/`.

Current setup:

- train dataset is required
- eval dataset is required
- `remove_unused_columns=False`
- evaluation runs every epoch
- checkpoint saving runs every epoch
- default reporting target is `tensorboard`
- mixed precision (`fp16`) is enabled automatically when CUDA is available

Metrics are computed in the evaluation layer under `src/`:

- `CER` (primary)
- `WER` (secondary)

## Inference

Inference uses greedy CTC decoding:

1. tokenize the Hebrew input
2. run the model
3. take `argmax` over logits
4. collapse repeated tokens
5. remove CTC blank tokens
6. decode token ids back into the final IPA string

## Known Implementation Detail

The model `dicta-il/dictabert-large-char-menaked` may load as a custom `BertForDiacritization` wrapper instead of a plain `BertModel`.

The project handles this by checking `hasattr(model, "bert")` and using `model.bert` when present.

## Future Extensions

- add corpus-wide symbol audits before large preprocessing runs
- optionally support permissive ingestion with `UNK`
- add checkpoint export / better inference packaging
- add stronger evaluation tooling if `CER` and `WER` are not sufficient

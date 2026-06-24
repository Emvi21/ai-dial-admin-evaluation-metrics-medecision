# VCR Cassettes for Integration Tests

Integration tests need to make LLM requests to DIAL. To avoid making real LLM calls on every test run (slow, requires credentials, costs money), we record DIAL requests/responses once and replay them.

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) (based on [VCR.py](https://vcrpy.readthedocs.io/)) to record HTTP interactions. Cassettes are YAML files containing recorded requests/responses that allow tests to run offline without real API credentials.

## Directory Structure

pytest-recording creates a `cassettes/` directory next to each test file:

```
tests/
  metrics/
    aidial_rag_eval/
      test_nli_integration.py
      cassettes/                              # Auto-created
        test_nli_integration/                 # Named after test module
          test_metric_basic_evaluation.yaml   # Named after test function
```

## Adding a New Integration Test

1. **Mark your test with decorators:**
   ```python
   @pytest.mark.integration
   @pytest.mark.vcr
   def test_my_integration(test_dial_config):
       llm_factory = create_llm_factory(test_dial_config)
       # ... your test code using real LLM calls
   ```

2. **Configure DIAL credentials in `.env` file:**
   ```bash
   # Copy .env.example to .env if you haven't already
   cp .env.example .env
   
   # Edit .env and set your DIAL instance:
   DIAL_URL=https://your-dial-instance.example.com
   DIAL_API_KEY=your-actual-api-key
   ```

3. **Record cassettes:**
   ```bash
   make test ARGS="-m integration"
   ```

4. **Cassette is created automatically** at:
   `tests/your_module/cassettes/test_your_module/test_my_integration.yaml`

5. **Commit the cassette** - URLs and credentials are automatically normalized/redacted

## Running Integration Tests

**With cassettes (no credentials needed):**
```bash
make test ARGS="-m integration"
```

**Skip integration tests:**
```bash
make test ARGS="-m 'not integration'"
```

## Updating Cassettes

When API changes or you need fresh recordings:

```bash
# Ensure .env has valid DIAL credentials (see above)

# Re-record all integration tests
make test ARGS="--record-mode=rewrite -m integration"

# Re-record specific test file
make test ARGS="--record-mode=rewrite tests/metrics/aidial_rag_eval/test_nli_integration.py"
```

## Key Features

- **URL normalization**: Your DIAL URL → `dial.test.example.com` in cassettes
- **Credential filtering**: API keys automatically redacted
- **Readable format**: Response bodies are plain JSON (not gzipped)
- **Selective recording**: Only DIAL API calls recorded (NLTK, GitHub etc. excluded)

Configuration is in [`tests/conftest.py`](conftest.py) (`vcr_config` fixture).

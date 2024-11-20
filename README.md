# How to use
1. Create and activate your virtual environment `python -m venv venv`.
2. Download all dependencies `pip install -r requirements.txt`.
3. Download your LLM, put in pun_detection folder. I utilized Llama 3.2-3B as a GGUF from [here](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF).
4. Create and fill `.env` file with necessary API keys
5. Download and install rasa-pro: [instructions](https://rasa.com/docs/rasa-pro/installation/python/installation), [licensing](https://rasa.com/docs/rasa-pro/installation/python/licensing)
6. Train and run rasa `rasa train`, `rasa run`

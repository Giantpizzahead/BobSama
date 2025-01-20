# BobSama

Real-time Discord voice chatbot with screenshare interaction, low latency, and no-code setup.

## Setup

Use **Python 3.12** in a virtual environment! You'll also need a `GOOGLE_API_KEY` in a `.env` file.

```sh
# Make sure you are in a Python 3.12 virtual environment first
pip install pip-tools
pip-sync requirements.txt
```

After updating `requirements.in`:

```sh
pip-compile requirements.in --strip-extras
pip-sync requirements.txt
```

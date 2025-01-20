# BobSama

Real-time voice chat bot with screenshare interaction, V-tuber model, captions, and no-code setup.

## Setup

Use **Python 3.12** in a virtual environment!

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

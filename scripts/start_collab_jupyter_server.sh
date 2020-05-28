#!/usr/bin/env bash

# Allow web socket connections to the server:

jupyter serverextension enable --py jupyter_http_over_ws

# Start the server:
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0

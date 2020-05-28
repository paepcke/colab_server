#!/usr/bin/env bash

# Allow web socket connections to the server:

# Using setup.py installation for google-colab fails,
# b/c IPython versions conflict. However, pip installing
# directly works. Check whether already installed:

MODULE_EXISTS=$(pip list | grep google-colab)
if [[ -z $MODULE_EXISTS ]]
then
    echo "Installing google-colab..."
    pip install google-colab
    echo "Installing google-colab maybe done."    
fi


jupyter serverextension enable --py jupyter_http_over_ws

# Start the server:
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0

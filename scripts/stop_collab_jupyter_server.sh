#!/usr/bin/env bash

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )
PIDFILE=$SCRIPT_DIR/colab_server.pid

if [[ ! -f $PIDFILE ]]
then
    echo "No pid file; cannot kill."
    exit 1
fi

# Kill jupyter server
kill -9 $(<"$PIDFILE")


PORT_LISTENER=$(lsof -i :8888)

if [[ -z $PORT_LISTENER ]]
then
    echo "Killed successfully"
else
    echo "Kill failed."
    exit 1
fi


#!/usr/bin/env bash

SERVER_PORT=8888

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )
PIDFILE=$SCRIPT_DIR/colab_server.pid

if [ -f "$PIDFILE" ]
then
    PID=$(<"$PIDFILE")
else
    echo "No pid file; cannot tell."
    exit 1
fi

# Get something like:
#    COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME ZMQbg/3 102450 paepcke 5u ...
PORT_STATUS_LINE=$(lsof -i :$SERVER_PORT)
# Grab col 11, the PID:
PID_ON_PORT=$(echo $PORT_STATUS_LINE | cut -d" " -f11)

# PID on the port same as known server PID?
if [[ $PID_ON_PORT == $PID ]]
then
    echo "Server alive"
else
    echo "Server not running"
fi

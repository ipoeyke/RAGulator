#!/bin/bash
source ragulator-env/bin/activate
python ragulator/app.py &
APP_PID=$!

# define explicit cleanup
cleanup() {
    echo "Stopping app (PID $APP_PID)..."
    kill $APP_PID
    ray stop --force
    exit 0
}

# wait for SIGINT/SIGTERM and trap
trap cleanup SIGINT SIGTERM
wait $APP_PID
#!/bin/bash
set -e
cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"/

echo "Starting server"
OMP_NUM_THREADS=4 python server.py &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in $(seq 0 9); do
    echo "Starting client $i"
    OMP_NUM_THREADS=1 python client.py --node-id "${i}" &
done

# This will allow you to use CTRL+C to stop all background processes
trap 'trap - SIGTERM && kill -- -$$' SIGINT SIGTERM
# Wait for all background processes to complete
wait

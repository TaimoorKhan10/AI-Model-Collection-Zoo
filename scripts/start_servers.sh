#!/bin/bash

# Start the API server in the background
python -m api.app &

# Start the Webapp server in the foreground
python -m webapp.app

# Wait for background processes to finish (though webapp.app runs in foreground)
wait
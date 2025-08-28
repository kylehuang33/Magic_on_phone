#!/bin/bash

SESSION="ai_servers"

# Start new tmux session (detached)
tmux new-session -d -s $SESSION

# First window: llama-server
tmux rename-window -t $SESSION:0 'llama-server'
tmux send-keys -t $SESSION:0 "cd ~/models/gemma-3-1b-it-GGUF && llama-server -m gemma-3-1b-it-Q4_K_M.gguf --port 8080 -c 2048 -ngl 35" C-m

# Second window: ollama
tmux new-window -t $SESSION:1 -n 'ollama'
tmux send-keys -t $SESSION:1 "ollama serve" C-m

# Attach to the session
tmux attach -t $SESSION

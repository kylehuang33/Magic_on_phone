#!/bin/bash

# If no commit message is given, use "auto commit"
COMMIT_MSG=${1:-"minor changes"}

git add .
git commit -m "$COMMIT_MSG"
git push

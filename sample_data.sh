#!/bin/bash
# fetch sample dataset from https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt

echo "Downloading Shakespear dataset"

curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

echo "Dowload complete"
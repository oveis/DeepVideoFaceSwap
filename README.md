# How To setup and run the project

## Setup
`pip install -r requirements.txt`

## Collect data
`python3 faceswap.py preprocess --celebrity 'george clooney' --output-dir 'dataset' --limit 10`

## Train
`python3 faceswap.py train --trainer-name 'original' --batch-size 10 --iterations 10 --input-A 'dataset/faces/george clooney/frontal' --input-B 'dataset/faces/barack obama/frontal' --model-dir 'dataset/models' --num-gpu 1 --log 'debug'`
# How To setup and run the project

## Setup
`pip install -r requirements.txt`

## Collect data
`python faceswap.py extract -i ./download/images/george_clooney -o ./download/images/george_clooney_out/`
`python faceswap.py extract -i ./download/images/chi/chi_1.mov -o ./download/images/chi_out/`

## Train
`python faceswap.py train -A ./download/images/george_clooney_out/ -B ./download/images/liu_ye_out/ -m ./output/model/george_liuye/`

## Convert
`python faceswap.py convert -i ./download/images/george_clooney/ -o ./output/converted/george/ -m ./output/model/george_liuye/`
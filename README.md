# Deep Face Swap with GAN

## Contributors
* Jinil Jang
* Chi Wang

## Document
http://cs230.stanford.edu/projects_spring_2019/reports/18681213.pdf

## Poster
http://cs230.stanford.edu/projects_spring_2019/posters/18681474.pdf


# How To setup and run the project

## Setup
`pip install -r requirements.txt`

## Prepare dataset
There are 2 ways to prepare dataset; `Download images from google` or `Download pre-collected images from S3`

### Download from S3
`cd ~/DeepVideoFaceSwap && mkdir download && aws s3 cp --recursive s3://faceswap-dataset ./download`

### Download from google
`python faceswap.py collect -o "download" -k "george_clooney" --limit 100`

If you want to download images more than 100, you need to install Chrome. Follow this:
* Linux
  * Install Chrome: `curl https://intoli.com/install-google-chrome.sh | bash`
  * Download Chromedriver: `cd ~/DeepVideoFaceSwap && wget https://chromedriver.storage.googleapis.com/75.0.3770.8/chromedriver_linux64.zip && unzip chromedriver_linux64.zip && rm chromedriver_linux64.zip`
  
`python faceswap.py collect -o "download" -k "george_clooney" --limit 2000 --driver-path "~/DeepVideoFaceSwap/chromedriver"`

## Extract Faces from Images
`python faceswap.py extract -i ./download/images/george_clooney -o ./download/images/george_clooney_out/`
`python faceswap.py extract -i ./download/images/jinil_jang -o ./download/images/jinil_jang_out/`

## Train
`python faceswap.py train -A ./download/images/george_clooney_out/ -B ./download/images/jinil_jang_out/ -m ./output/model/george_jinil/`

## Convert
`python faceswap.py convert -i ./download/images/george_clooney/ -o ./output/converted/george_jinil/ -m ./output/model/george_jinil/`

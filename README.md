# How To setup and run the project

## Setup
`pip install -r requirements.txt`

## Collect Images from internet
`python faceswap.py collect -o "download" -k "george_clooney" --limit 100`

If you want to download images more than 100, you need to install Chrome. Follow this:
* Linux
  * Install Chrome: `curl https://intoli.com/install-google-chrome.sh | bash`
  * Download Chromedriver: `cd ~/DeepVideoFaceSwap && wget https://chromedriver.storage.googleapis.com/75.0.3770.8/chromedriver_linux64.zip && unzip chromedriver_linux64.zip && rm chromedriver_linux64.zip`
  
`python faceswap.py collect -o "download" -k "george_clooney" --limit 2000 --driver-path "~/DeepVideoFaceSwap/chromedriver"`

## Extract Faces from Images
`python faceswap.py extract -i ./download/images/george_clooney -o ./download/images/george_clooney_out/`
`python faceswap.py extract -i ./download/images/liu_ye -o ./download/images/liu_ye_out/`

## Train
`python faceswap.py train -A ./download/images/george_clooney_out/ -B ./download/images/liu_ye_out/ -m ./output/model/george_liuye/`

## Convert
`python faceswap.py convert -i ./download/images/george_clooney/ -o ./output/converted/george/ -m ./output/model/george_liuye/`

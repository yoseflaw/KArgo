python -m nltk.downloader stopwords
python -m nltk.downloader universal_tagset
python -m spacy download en
# install sent2vec
git clone https://github.com/epfml/sent2vec
cd sent2vec || exit
git checkout f827d014a473aa22b2fef28d9e29211d50808d48
pip install cython
cd src || exit
python setup.py build_ext
pip install .
#cd ../.. || exit
# download pretrained model and put it inside "pretrain_models" folder
# install EmbedRank (ai-research-keyphrase-extraction)
#git clone https://github.com/swisscom/ai-research-keyphrase-extraction.git
cd ai-research-keyphrase-extraction || exit
pip install -r requirements.txt
python -m nltk.downloader punkt
#  move swisscom_ai folder to root

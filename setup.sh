# Prepare directories
mkdir -p data/{hdfs_tokenized,hdfs_bert}
mkdir -p data/torch_dataset/{train,test}

# Download HDFS dataset
wget -P data "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
unzip data/HDFS_v1.zip

# Install dependencies
poetry install
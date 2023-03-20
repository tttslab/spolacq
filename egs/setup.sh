# Download dataset
if [ ! -d data ]
then
    wget https://tslab2.ip.titech.ac.jp/spolacq/data.zip
    unzip -q data.zip
fi

# Download pretrained speech-to-unit model
if [ ! -d models/S2U/models ]
then
    wget https://tslab2.ip.titech.ac.jp/spolacq/models.zip
    unzip -q models.zip
fi
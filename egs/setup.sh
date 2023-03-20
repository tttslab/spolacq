# Download dataset
if [ ! -d ../data ]
then
    cd ..
    wget https://tslab2.ip.titech.ac.jp/spolacq/data.zip
    unzip -q data.zip
    cd -
fi
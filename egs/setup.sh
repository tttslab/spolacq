# Download dataset
if [ ! -d ../data ]
then
    cd ..
    wget https://tslab2.ip.titech.ac.jp/spolacq/data.zip
    unzip -q data.zip
    cd -
fi

# Install third party libraries
if [ ! -d ../tools/recipe_zs2017_track2 ]
then
    cd ../tools
    git clone https://github.com/kamperh/recipe_zs2017_track2.git
    cd -
fi

if [ ! -d ../tools/eskmeans ]
then
    cd ../tools
    git clone https://github.com/kamperh/eskmeans.git
    cd -
fi

cp -r ../tools/recipe_zs2017_track2/syllables/thetaOscillator/ ../tools/syllables
cp -r ../tools/recipe_zs2017_track2/syllables/thetaOscillator/gammatone/gammatone_c.c ../tools/syllables/thetaOscillator
cp -r ../utils/eskmeans_api.py ../tools/eskmeans
mv ../tools/eskmeans/eskmeans_api.py ../tools/eskmeans/eskmeans_wordseg.py

cd ../tools/syllables/thetaOscillator
# Compile C file
matlab -batch "mex gammatone_c.c"
# Patch
patch process_wavs.m process_wavs.patch
cd -
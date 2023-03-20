# train speech-to-unit model from scratch
# cd ../tools/S2U
# bash scripts/preprocess.sh
# bash scripts/train.sh 01000 RDVQ_01000 ""
# bash scripts/train.sh 01100 RDVQ_01000_01100 "--seed-dir ./exps/RDVQ_01000"

# cd ../../models/S2U/models
# ln -s ../../../tools/S2U/exps/RDVQ_01000_01100/models/best_audio_model.pth
# cd ..
# ln -s ../../tools/S2U/exps/RDVQ_01000_01100/args.pkl

# download pretrained speech-to-unit model
# wget https://tslab2.ip.titech.ac.jp/path/to/pretrained_model
# unzip -q hoge.zip

cd ../../tools/I2U
python create_input_files.py
python train_i2u.py

cd ../../tools/U2S
python preprocess.py
python train.py --output_directory=../../models/U2S/outdir --log_directory=logdir

cd ../../utils
python wav2vec2_api.py

python main_spolacq3.py  # reinforcement learning
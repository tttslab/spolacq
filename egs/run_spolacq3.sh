cd tools/I2U
python create_input_files.py
python train_i2u.py

cd ../../tools/U2S
python preprocess.py
python train.py --output_directory=../../models/U2S/outdir --log_directory=logdir

cd ../../utils
python wav2vec2_api.py  # finetune wav2vec2
python main_spolacq3.py  # reinforcement learning
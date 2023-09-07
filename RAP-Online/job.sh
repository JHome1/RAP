DATASET=miniImagenet # [miniImagenet, CUB]
MODEL=Conv4_rein     # [Conv4_rein, Conv6_rein]
METHOD=maml          # [maml, protonet]
N_SHOT=1             # [1, 5]
# Choose the correspoding pre-trained base model

## Train
python ./train.py --dataset &DATASET --model &MODEL --method &METHOD --n_shot &N_SHOT --train_aug --PATH-base-model './pre_model/miniImageNet/maml/conv4_1shot_base_model.tar'

## Save features (only for protonet)
# python ./save_features.py --dataset &DATASET --model &MODEL --method protonet --n_shot &N_SHOT --train_aug --PATH-base-model './pre_model/miniImageNet/protonet/conv4_1shot_base_model.tar'

## Test
python ./test.py --dataset &DATASET --model &MODEL --method &METHOD --n_shot &N_SHOT --train_aug --PATH-base-model './pre_model/miniImageNet/maml/conv4_1shot_base_model.tar' 

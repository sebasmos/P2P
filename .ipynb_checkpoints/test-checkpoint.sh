echo "python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name saver  --results_dir results_adapter_augmented_horiz_v4_500_epochs --dataset_mode adapter  --direction BtoA "

python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name exp3_1  --results_dir exp3 --dataset_mode test  --direction BtoA 


#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name vcg_augmented_horiz_v3_500_epochs  --results_dir saver --dataset_mode test  --direction BtoA 

#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/test --model pix2pix --name adapter_augmented_horiz_v4_500_epochs  --results_dir results_adapter_augmented_horiz_v4_500_epochs --dataset_mode adapter  --direction BtoA  # --output_nc 19 #--input_nc 3  #--direction BtoA # --netG resnet_9blocks #

#python test.py --dataroot /n/pfister_lab2/Lab/scajas/DATASETS/DATASET_pix2pix/ --direction BtoA --model pix2pix --name vcg_changing_deconv_by_upsam --use_wandb --results_dir borrar1
DATASET='imagenetc_brightness imagenetc_contrast imagenetc_defocus_blur 
          imagenetc_elastic_transform imagenetc_fog imagenetc_frost 
          imagenetc_gaussian_blur imagenetc_gaussian_noise imagenetc_glass_blur 
          imagenetc_impulse_noise imagenetc_jpeg_compression imagenetc_motion_blur 
          imagenetc_pixelate imagenetc_saturate imagenetc_shot_noise 
          imagenetc_snow imagenetc_spatter imagenetc_speckle_noise imagenetc_zoom_blur'

SEVERITY='1 2 3 4 5'

# ResNet18
for dataset in $DATASET; do
	for severity in $SEVERITY; do
		python3 bn_adaptation.py \
			--adaptation_dataset $dataset  \
			--evaluation_dataset $dataset \
			--evaluate_before_adaptation \
			--severity $severity \
			--adapt_bn True \
			--adaptation_batch_size 32 \
			--dataset imagenet \
			--arch resnet18 \
			--batch_size 128 \
			--workers 4 
	done
done

DATASET='cifar100c_gaussian_noise cifar100c_speckle cifar100c_shot cifar100c_impulse 
          cifar100c_contrast cifar100c_elastic cifar100c_pixelate cifar100c_jpeg cifar100c_saturate 
          cifar100c_snow cifar100c_fog cifar100c_brightness cifar100c_defocus
          cifar100c_frost cifar100c_spatter cifar100c_glass cifar100c_motion cifar100c_zoom cifar100c_gaussian_blur'

SEVERITY='1 2 3 4 5'

# ResNet20
for dataset in $DATASET; do
	for severity in $SEVERITY; do
		python3 bn_adaptation.py \
			--adaptation_dataset $dataset \
			--evaluation_dataset $dataset \
			--evaluate_before_adaptation \
			--severity ${severity} \
			--adapt_bn True \
			--adaptation_batch_size 32 \
			--dataset cifar100 \
			--arch resnet20_cifar100 \
			--batch_size 128 \
			--workers 4
	done
done

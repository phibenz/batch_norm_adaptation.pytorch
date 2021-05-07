DATASET='cifar10c_gaussian_noise cifar10c_speckle cifar10c_shot cifar10c_impulse 
          cifar10c_contrast cifar10c_elastic cifar10c_pixelate cifar10c_jpeg cifar10c_saturate 
          cifar10c_snow cifar10c_fog cifar10c_brightness cifar10c_defocus
          cifar10c_frost cifar10c_spatter cifar10c_glass cifar10c_motion cifar10c_zoom cifar10c_gaussian_blur'

SEVERITY='1 2 3 4 5'

# ResNet20
for dataset in $DATASET; do
	for severity in $SEVERITY; do
		python3 bn_adaptation.py \
			--adaptation_dataset $dataset  \
			--evaluation_dataset $dataset \
			--evaluate_before_adaptation \
			--severity ${severity} \
			--adapt_bn True \
			--adaptation_batch_size 32 \
			--dataset cifar10 \
			--arch resnet20_cifar10 \
			--batch_size 128 \
			--workers 4
	done
done

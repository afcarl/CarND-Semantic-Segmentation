PHONY: create-conda remove-conda

NAME := carnd-semantic-segmentation

create-conda:
	conda env create -f environment.yml -n $(NAME)

remove-conda:
	conda env remove -y -n $(NAME)

download-vgg:
	curl -O https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip

download-data:
	curl -O http://kitti.is.tue.mpg.de/kitti/data_road.zip

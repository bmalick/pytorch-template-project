help:
	@echo "you can do it !"

	# export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python-path:
	@export PYTHONPATH=$(PYTHONPATH):$(shell pwd)/src

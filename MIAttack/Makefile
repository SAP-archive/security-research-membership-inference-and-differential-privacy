LOG_PATH=./logs
FOLDER := data logs models experiments
TARGETS := install uninstall clean folders download_vgg16

.PHONY: $(TARGETS)

all: install

folders:
	mkdir -p $(FOLDER)

install: folders
	echo "Installing mia..."
	python -m pip install -r ./requirements.txt
	python -m pip install -e ./

download_vgg16: folders
	curl -o models/rcmalli_vggface_tf_notop_vgg16.h5 -L https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_vgg16.h5

uninstall:
	pip uninstall mia

clean:
	python -m pip uninstall mia
	python -m pip uninstall -r requirements.txt
	rm -rf $(FOLDER)

test:
	pytest ./mia/test

PROJECT_ID="simple_diffusion"
IMAGE_ID="simple_diffusion_image"
VERSION_ID="v1.0"
IMAGE_URI="gcr.io/$(PROJECT_ID)/$(IMAGE_ID):$(VERSION_ID)"

build:
	docker build . -t $(IMAGE_URI)
run:
	docker run $(IMAGE_URI)
push:
	docker push $(IMAGE_URI)
env-create:
	mamba create --name $(PROJECT_ID) python=3.8 --file req/req.txt
env-activate:
	mamba activate $(PROJECT_ID)
env-save:
	mamba list -e > req/req.txt
PROJECT_ID="new_project"
IMAGE_ID="new_project_image"
VERSION_ID="v1.0"
IMAGE_URI="gcr.io/$(PROJECT_ID)/$(IMAGE_ID):$(VERSION_ID)"

build:
	docker build . -t $(IMAGE_URI)
run:
	docker run $(IMAGE_URI)
push:
	docker push $(IMAGE_URI)

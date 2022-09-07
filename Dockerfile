FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-11

COPY . /workspace
WORKDIR /workspace

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "-m", "train"]

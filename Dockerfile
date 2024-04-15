FROM nvcr.io/nvidia/pytorch:24.03-py3


WORKDIR /dl-container

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt 

RUN pip install --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON lightgbm

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
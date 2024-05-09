FROM nvcr.io/nvidia/pytorch:24.04-py3


WORKDIR /dl-container

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt 

RUN pip uninstall lightgbm -y

RUN pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON lightgbm

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
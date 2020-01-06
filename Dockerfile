FROM tensorflow/tensorflow:latest-gpu-py3



RUN apt-get update && \
	apt-get install -y libsm6 libxext6 libxrender-dev && \
	pip install --upgrade pip && \
	pip install opencv-python && \
	pip install pandas && \
	pip install tqdm && \
	pip install -vU setuptools && \
	pip install jupyter && \
	pip install pydicom && \
	pip install seaborn

RUN apt-get update && \ 
     apt-get install -yq --no-install-recommends \ 
     libasound2 libatk1.0-0 libc6 libcairo2 libcups2 libdbus-1-3 \ 
     libexpat1 libfontconfig1 libgcc1 libgconf-2-4 libgdk-pixbuf2.0-0 libglib2.0-0 libgtk-3-0 libnspr4 \ 
     libpango-1.0-0 libpangocairo-1.0-0 libstdc++6 libx11-6 libx11-xcb1 libxcb1 \ 
     libxcursor1 libxdamage1 libxext6 libxfixes3 libxi6 libxrandr2 libxrender1 libxss1 libxtst6 \ 
     libnss3 libgl1-mesa-glx

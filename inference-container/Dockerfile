FROM iaghidro/inference-container:latest

# Set up the program in the image
COPY conv_net /opt/program
WORKDIR /opt/program
RUN chmod 755 serve

RUN ln -s /fastai/fastai fastai
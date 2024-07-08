# Create a Streamlit app
FROM ubuntu
WORKDIR /app
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get install -y build-essential curl software-properties-common git unzip
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get install -y python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/*
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && unzip awscliv2.zip
RUN ./aws/install
RUN rm -rf ./aws awscliv2.zip
COPY requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["streamlit", "run", "Home.py", "--server.port=5000", "--server.address=0.0.0.0"]
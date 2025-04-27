#Use tensorflow 2.7.0 as the base image
FROM tensorflow/tensorflow:2.7.0-gpu

#Install the dependencies
RUN pip install pyitlib pgmpy importlib_resources ucimlrepo matplotlib

#Install the ssh
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && apt-get update
RUN apt install openssh-server wget vim -y

#Start the service
RUN service ssh start

#Copy Ganblr 0.1.1 folder
#COPY ganblr-0.1.1 /ganblr

#Set the WORKDIR
#WORKDIR /ganblr

##Download the ganblr code
#RUN wget https://files.pythonhosted.org/packages/c2/a6/e4097efcdcff218e5a2134ad06e633b93352e56087a22e5840267aca920b/ganblr-0.1.1.tar.gz
#RUN tar -xzvf ganblr-0.1.1.tar.gz

EXPOSE 22 6006
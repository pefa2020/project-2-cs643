# How to run

- To create the environment first create and connect into an EC2 instance : Amazon Linux t3-medium

- Next clone this repository in that EC2:
    - git clone https://github.com/pefa2020/project-2-cs643.git
    - cd project-2-cs643

- WITHOUT DOCKER approach on EC2 Instance:
    -  python3 -m venv .venv
    - source .venv/bin/activate
    - pip install -r requirements.txt
    - sudo dnf install -y java-21-amazon-corretto
    - spark-submit predict.py ValidationDataset.csv

- WITH DOCKER approach on EC2 instance:
    - sudo yum install docker
    - sudo systemctl start docker
    - sudo systemctl enable docker
    - Run the command below to obtain F1 score. Note that you can run a path to a csv file instead of ValidationDataset.csv:
        - sudo docker run pefa2020/wine-predictor ValidationDataset.csv

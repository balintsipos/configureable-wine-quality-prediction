FROM python:3.10

ADD main.py .
ADD winequality.csv .

RUN pip install matplotlib seaborn scikit-learn pandas

ENTRYPOINT ["python", "main.py" ]
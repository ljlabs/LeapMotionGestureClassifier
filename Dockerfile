FROM python:3.9

RUN pip install fastapi uvicorn
RUN pip install --upgrade tensorflow
RUN pip install numpy==1.19.2

EXPOSE 8000

COPY ./LeapMotionGestureRecognizer /app
WORKDIR /app

CMD python main.py
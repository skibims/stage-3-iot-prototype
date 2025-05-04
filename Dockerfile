FROM python:3.13-slim
WORKDIR /app
COPY . /app
WORKDIR /app/backend
RUN pip install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5000
CMD ["gunicorn", "Model:app", "--bind", "0.0.0.0:5000"]
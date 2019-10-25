# Code Examples for the TensorFlow World talk on "Advanced model deployments with TensorFlow Serving"

This repository contains all code examples for my TensorFlow World talk about "Advanced model deployments with TensorFlow Serving"

If you have questions, please reach out via twitter @hanneshapke.
You can also find further information on [buildingmlpipelines.com](http://buildingmlpipelines.com).


### Model Data Set

The example model was trained with the dataset for "IMDB movie reviews for Sentiment Analysis" which can be found [here](https://www.kaggle.com/oumaimahourrane/imdb-reviews/data). In order to retrain the model, log in to Kaggle and obtain the dataset. It is assumed that the uncompressed file is saved in the same folder as `train.py` file.

For copyright reasons, the data set isn't included in this repository.


### Getting started

```
$ virtualenv -p python3 tf-world-tf-serving
$ source tf-world-tf-serving/bin/activate
$ pip3 install -r requirements.txt
```



### Starting up TensorFlow Serving


#### 90 sec Model Deployment

With the example below you can start your TensorFlow Serving instance on your host machine.

```
$ docker run -p 8500:8500 \
             -p 8501:8501 \
    		 --mount type=bind, source=saved_models/, target=/models/my_model \
             -e MODEL_NAME=my_model
             -t tensorflow/serving
```

#### Loading two Versions of the same Model

The example below lets you server multiple models. You'll need to update your configuration file before starting the server.

```
docker run -p 8501:8501 \
    --mount type=bind,source=`pwd`,target=/models/my_model \
    --mount type=bind,source=`pwd`/../../example_tf_serving_configurations,target=/models/model_config \
    -t tensorflow/serving \
    --model_config_file=/models/model_config/model_config_list.txt
```

#### Loading two Versions of the same Model with Version Labels

The example below lets you server multiple model version. You'll need to update your configuration file before starting the server.

```
docker run -p 8501:8501 \
    --mount type=bind,source=`pwd`,target=/models/my_model \
    --mount type=bind,source=`pwd`/../../example_tf_serving_configurations,target=/models/model_config \
    -t tensorflow/serving \
    --model_config_file=/models/model_config/model_config_list_with_labels.txt \
    --allow_version_labels_for_unavailable_models=true
```



#### Run Prometheus Container

With the Prometheus config file below, you can run a basic instance of Prometheus.

```
global:
  scrape_interval:     15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'tf-serving-monitor'
scrape_configs:
  - job_name: 'prometheus'
    scrape_interval: 5s
    metrics_path: /monitoring/prometheus/metrics
    static_configs:
      - targets: ['host.docker.internal:8501']
```

```
docker run -p 9090:9090 -v /tmp/prometheus.yml:/etc/prometheus/prometheus.yml \
       prom/prometheus
```

Once Prometheus is running, you can config TFServing to provide an monitoring endpoint for Prometheus.

```
docker run -p 8501:8501 \
    --mount type=bind,source=`pwd`,target=/models/my_model \
    --mount type=bind,source=`pwd`/../../example_tf_serving_configurations,target=/models/model_config \
    -t tensorflow/serving \
    --model_config_file=/models/model_config/model_config_list_with_labels.txt \
    --monitoring_config_file=/models/model_config/monitoring_config_file.txt
```

#### Loading TF Lite Models

If you want to host TFLite models, use the command below to run your TFServing instance.

```
docker run -p 8501:8501 \
    --mount type=bind,source=`pwd`/tflite,target=/models/my_model \
    -e MODEL_BASE_PATH=/models \
    -e MODEL_NAME=my_model \
    -t tensorflow/serving:latest \
    --use_tflite_model=true
```

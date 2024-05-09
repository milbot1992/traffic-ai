# Traffic Sign Classification

## Overview
This project implements a traffic sign classification system using convolutional neural networks (CNNs). The system includes a script for training a model on traffic sign images and a Flask application to serve the model for classifying images via HTTP requests.

## Project Structure
- `traffic.py`: Trains a CNN on a dataset of traffic sign images and saves the model.
- `app.py`: Flask application that uses the trained model to classify images uploaded via HTTP.

## Prerequisites
- Python 3.x
- Flask
- TensorFlow
- NumPy
- OpenCV-Python

## Setup and Installation
1. **Clone the Repository**:
    ```
    git clone https://github.com/your-repository/traffic-classification.git
    cd traffic-classification
    ```

2. **Install Dependencies**:
    - It's recommended to use a virtual environment:
        ```
        python -m venv env
        source env/bin/activate  # On Windows use `env\Scripts\activate`
        ```
    - Install required packages:
        ```
        pip install -r requirements.txt
        ```

3. **Prepare the Dataset**:
    - Ensure your dataset is structured with images categorized into folders named `0` to `42`, corresponding to different traffic signs.
    - Place your dataset directory in the project root or specify the path when running the training script.

## Training the Model
Run `traffic.py` with the path to your dataset to train the model and save it:

```python traffic.py /path/to/data_directory```

- **Arguments**:
  - `data_directory`: Path to the dataset directory.
  - `optional_model_name`: Optional. If specified, the trained model will be saved with this name. Default is `model.h5.keras`.

## Running the Flask Application
After training, start the Flask server to classify images:

```python app.py```

The server will run on `http://localhost:5000/`.

## API Usage
- **Endpoint**: `/classify`
- **Method**: POST
- **Payload**: Multipart/form-data with an image file under the key `file`.
- **Response**: JSON containing the predicted class of the traffic sign.

## Example Curl Request
```curl -X POST -F 'file=@path_to_image.jpg' http://localhost:5000/classify```


## Contributing
Contributions to this project are welcome. Please follow the standard fork-pull request workflow.

## License
This project is licensed under the MIT License

# Cognitive Fatigue Prediction and Dashboard

## Overview
This software is developed as part of a dissertation project to address cognitive fatigue, a significant decrease in mental performance due to sustained mental workload. The project features two main components:
1. **Cognitive Fatigue Prediction**: Utilizing machine learning techniques such as K-Means Clustering, Decision Tree, Artificial Neural Network, and Support Vector Machine to predict cognitive fatigue.
2. **Conceptual Dashboard Development**: Implementing a user interface using Dash to visualize the predictions and insights derived from the machine learning models.

The Support Vector Machine model has shown the best performance with an average accuracy of 67% and a recall score of 62%. This tool aims to assist in improving productivity and performance in non-critical tasks by predicting cognitive fatigue.

## Installation

### Prerequisites
- **Anaconda**: The project requires Anaconda to manage dependencies and environments. Install Anaconda for your operating system using the following links:
  - [Anaconda for macOS](https://docs.anaconda.com/anaconda/install/mac-os/)
  - [Anaconda for Windows](https://docs.anaconda.com/anaconda/install/windows/)

### Setup
1. **Download the Project**: Download and unzip the project files to your local machine.
2. **Environment Setup**:
   Open a terminal or command prompt and navigate to the project directory where the `environment.yml` file is located.
   ```bash
   conda env create -f environment.yml

### 3. Activate the Environment:
```bash
conda activate project_env
```

### Launch the Dashboard:
```bash
python app.py
```
After launching the app, open a web browser and visit [http://127.0.0.1:8050/](http://127.0.0.1:8050/) to view the dashboard.

### Usage
- **Interacting with the Dashboard**: Use the dashboard to input data and receive predictions regarding cognitive fatigue.
- **Termination**: To stop the dashboard, press `Ctrl + C` in the terminal.

### Uninstallation
To uninstall Anaconda and clean up all dependencies, follow the instructions here: [Uninstall Anaconda](https://docs.anaconda.com/anaconda/install/uninstall/).

### License
[MIT License](LICENSE)

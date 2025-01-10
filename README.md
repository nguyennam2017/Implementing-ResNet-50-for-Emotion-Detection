# Emotion Detection using ResNet-50

This project aims to detect emotions from facial images using a deep learning model based on the ResNet-50 architecture. The dataset used for training and testing the model is the FER-2013 dataset, which contains images categorized into seven emotion classes: angry, disgust, fear, happy, neutral, sad, and surprise.

## Dataset structure
- Data/
    - FER_2013/
        - test/
            - angry/
            - disgust/
            - fear/
            - happy/
            - neutral/
            - sad/
            - surprise/
        - train/
            - angry/
            - disgust/
            - fear/
            - happy/
            - neutral/
            - sad/
            - surprise/

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-learn

### Installation

1. Clone the repository:
    \`\`\`sh
    git clone https://github.com/yourusername/emotion-detection.git
    cd emotion-detection
    \`\`\`

2. Install the required packages (In each `.ipynb` file)

3. Download the FER-2013 dataset from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013/data) and place it in the \`Data/FER_2013\` directory.

## Usage

1. Open the Jupyter notebook \`ResNet50.ipynb\` or \`3-1-project-i_kaggle.ipynb\` in Visual Studio Code or Jupyter Notebook.
2. Run the cells to preprocess the data, build the model, train the model, and evaluate the model.
3. The trained model will be saved as \`models/emotion_detection_model.h5\`.

## Results

The training and validation accuracy and loss will be displayed during the training process. The final model's performance can be evaluated using the confusion matrix and accuracy metrics.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The FER-2013 dataset is provided by Kaggle.
- The ResNet-50 architecture is based on the original paper by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.

For more details, refer to the project report \`Report_project_I.pptx\`.
# Project: Texas Holdem Poker Odds Calculator
## Team Information

| Team Members       | Email Address   |
|--------------------|-----------------|
| Kevin Daniel       | kdanie4@gmu.edu |
| Saianurag Kommuri  | skommur@gmu.edu |
| Shafiullah Hashimi | shashim5@gmu.edu|
| Reza Khoshhal      | rkhoshha@gmu.edu|

## Motivation and Background

- This project aims to detect and identify playing cards on a Texas Holdem poker table using computer vision techniques and calculate the odds of having a winning hand.

### Goals

1. Gain more experience with feature detection and mapping methods.
2. Combine computer vision concepts with other high-level coding concepts.
3. Train a computer to predict the odds of winning at poker, providing an interesting and enjoyable project for the team.

## Project Specifications

### Dataset

- Utilizes the Playing Cards Object Detection Dataset from Kaggle.com
- Contains 20,000 images of playing cards in various lighting, orientations, and backgrounds

### Model

- Uses YOLOv5, an object detection model for bounding box prediction and object classification
- Training time is estimated to be around 8 hours for sufficient accuracy

### Poker Hand Evaluation

- Calculates the best possible hand based on the board and the hand's exact rank
- Employs a lookup table of 7,462 unique poker hand ranks for fast calculations
- Leverages the openly-available deuces Python module

### Outcome

- The completed poker program will be able to decide whether to bet or fold based on the calculated likelihood of the current hand winning.


## Environment Setup and Requirements

## Python Setup

We will be using Python 3 for this project. If you don't have Python 3 installed, you can install it based on your operating system:

- **Mac**: Using the Homebrew package manager, you can install Python 3 with `brew install python3`.
- **Linux**: On Debian systems, you can install Python 3 with `sudo apt install python3`.

After installing Python 3, we recommend setting up a Python virtual environment for the project. Here's how you can do it:

```bash
# Move to the project folder
cd $PROJECT_FOLDER

# Create a virtual environment and activate it
python3 -m venv cs482venv
source cs482venv/bin/activate
```
After setting up the environment, you can install the necessary libraries with the following command:
```bash
pip install -r requirements.txt
```
The `requirements.txt` file should include the following libraries in order for you to succesfully run the jupyter file:

To add the virtual environment to Jupyter Notebook, use the following command:
`python -m ipykernel install --user --name=cs482venv --display-name "Python3 (CS482)" `

## Running the Jupyter Notebook:
`jupyter notebook`

Please replace `$PROJECT_FOLDER` with the path to your project folder.

If you are still having trouble, The full environment setup document is available [here](./CS482%20Computer%20Vision%20-%20Python%20Getting%20Started.pdf).


## Data Sources

| Link | Description |
|------|-------------|
| [Link1](URL1) | Description of Link1 |
| [Link2](URL2) | Description of Link2 |
| [Link3](URL3) | Description of Link3 |

## Literature Survey

Below are some of the key literature and resources we referred to for this project:

| Title | Author(s) | Link | Key Findings |
|-------|-----------|------|--------------|
| Title1 | Author1 | [Link1](URL1) | Brief summary of key findings from source 1 |
| Title2 | Author2 | [Link2](URL2) | Brief summary of key findings from source 2 |
| Title3 | Author3 | [Link3](URL3) | Brief summary of key findings from source 3 |


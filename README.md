# Project: Texas Holdem Poker Odds Calculator
## Team Information

| Team Members       | Email Address   |
|--------------------|-----------------|
| Kevin Daniel       | kdanie4@gmu.edu |
| Saianurag Kommuri  | skommur@gmu.edu |
| Shafiullah Hashimi | shashim5@gmu.edu|
| Reza Khoshhal      | rkhoshha@gmu.edu|

## Project Proposal 
- The project proposal can be found [here](https://docs.google.com/document/d/15ecoP3ZsK5myAPOcKUsPFbqGT3KCkJQoNnVF8iDrlcQ/).

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
To run the jupyter file use `jupyter notebook` in your terminal. 

Please replace `$PROJECT_FOLDER` with the path to your project folder.

If you are still having trouble, The full environment setup document is available [here](./CS482%20Computer%20Vision%20-%20Python%20Getting%20Started.pdf).


## Data Sources

| Link | Description |
|------|-------------|
| [OpenCV Card Detector](https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector/blob/master/Cards.py) | Detection code for the cards was sourced here |
| [Poker Hand Image Recognition](https://samyzaf.com/ML/poker2/poker2.html) | The provided link leads to a webpage showcasing a machine learning model trained to detect poker hands from images. |
| [Poker ML](https://github.com/sagor999/poker_ml) | The provided link showcases a machine learning model designed to recognize and classify poker hands from images. |

## Literature Survey

Below are some of the key literature and resources we referred to for this project:

| Title | Author(s) | Link | Key Findings |
|-------|-----------|------|--------------|
| Rain Man 2.0 - Blackjack Robot | J. Grillo | [Blackjack Robot](https://hackaday.io/project/27639-rain-man-20-blackjack-robot.) | The project showcases a Raspberry Pi-based robot that can identify cards in its own hand and the dealer's hand, use a table to determine optimal plays, and employ card counting strategies. Although focused on blackjack, the card detection capability demonstrated with a 99% match rate provides insights applicable to our poker project.|
| A method of computing winning probability for Texas Hold'em poker| Z. Xiaochuan, D. Song, Z. Hailu, L. He, W. Fan | [Computing Probability for Hold'em]([URL2](http://www.ijmlc.org/papers/275-LC009.pdf)) | The study presents an algorithm for computing winning probability in Texas Hold'em poker. It involves linear regression to approximate effective hand strength and classification of 5 or 6 cards to generate a feature vector. The algorithm and techniques such as reinforcement learning (RL), Q-learning, and Deep Q-Networks have potential for improving decision-making and training accuracy in our project. |
| Image information and visual quality | H.R. Sheikh | [Image Recognition](https://ieeexplore.ieee.org/abstract/document/1576816?casa_token=DxQnaal5t3AAAAAA:wQq0qZWCt-_YJic-zXD-OxRedXb5AdJQtD1TspsTfOvGwCQKTly1e57sq51XhXep6tZXaWUI) | The provided link leads to an IEEE abstract discussing a method for hand strength evaluation in Texas Hold'em poker. The approach utilizes a machine learning algorithm trained on historical data to predict the winning probability of a poker hand, offering potential insights for our project. |

## Usage of ChatGPT in this Project

In the development of this project, we extensively used OpenAI's ChatGPT, specifically the GPT-4 model. Here's how it contributed:

1. **Concept Clarification**: Whenever we encountered complex concepts related to machine learning, computer vision, or game theory, ChatGPT provided clear and concise explanations, which helped us understand and implement these concepts effectively.

2. **Debugging Assistance**: ChatGPT was instrumental in helping us debug our code. When we encountered errors or issues, we consulted ChatGPT to get possible solutions or workarounds, which significantly expedited the debugging process.

3. **Code Improvement**: ChatGPT also offered suggestions to enhance our code. It provided insights into better coding practices and suggested more efficient or cleaner ways to write certain parts of our code.

4. **Project Documentation**: ChatGPT assisted us in documenting our project. It helped us formulate this README file and other necessary documentation, ensuring our project is understandable and accessible to others.

ChatGPT proved to be a powerful tool, enhancing the speed and quality of our project development. We highly recommend its use in similar projects.



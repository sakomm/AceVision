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


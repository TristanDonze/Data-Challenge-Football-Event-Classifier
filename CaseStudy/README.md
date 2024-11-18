# Data Challenge Football Event Classifier

## Overview
This is a case study for a football event classifier. It includes various charts to study features related to the challenge without using NLP. The charts are created using JavaScript and D3.js. You can also find a conclusion at the end, about what we can understand of this chart, what are the limitation and what could be interesting to do.

## Running the Project
The first step is to unzip the data in CaseStudy/UNZIP-ME.zip 
    --> CaseStudy/data/[challenge_data, eval_tweets, train_tweets]

To run the project, you need to start a simple HTTP server in the `CaseStudy` directory. You can do this using Python's built-in HTTP server.

### Steps to Run the Server
1. Open a terminal.
2. Navigate to the `CaseStudy` directory:
    ```sh
    cd ./CaseStudy
    ```
3. Start the HTTP server:
    ```sh
    python3 -m http.server
    ```
4. Open your web browser and go to `http://localhost:8000`.

## Warning
In the web interface, it is possible to display all the lines in the charts at the same time. **Do not do this** as it will consume a large amount of RAM and may crash your computer. Instead, display the lines one by one.

## Technologies Used
- JavaScript
- D3.js
- Python (for the HTTP server)

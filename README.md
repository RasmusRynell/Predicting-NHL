# Predicting NHL stats (shots on goal)

## Overview
This project explores the idea of using different machine learning techniques to determine different stats in NHL games. Research and testing of different techniques has previously been done in [this](https://github.com/RasmusRynell/sports_betting_test) project.

### Current features
- About 5000 predictions for “Shots on goal” in NHL games from different betting sites
- A way to add more "new" predictions by copy-pasting them from sites into text files
- Request data from NHL’s own database to populate your own internal database (in order to use less internet and have a faster running program)
- Produce a CSV file containing all data related to a prediction, this includes player statistics, players teams statistics, and enemy teams statistics for as many games back as one wants (typically no more than 5)

### Futures features
- Load and select features from CSV file to be used in different ML techniques (Data preprocessing)
- Create a pipeline for both using and testing and evaluating ML techniques
- Add more data (maybe xG for both player and teams)
- Create a better way of scraping the web both for data
- Create a better way of scraping the web both for previous bets and their outcomes
- Create a user interface either as an app or on the web

### Long term new features
- Create a more robust framework for different sports with different types of data
- Live predictions

<br><br/>
## Inner workings *(Under construction)*
### Data collection


### Preprocessing


### Machine learning


### Evaluating the results

<br><br/>
## Installation
*Side comment:
Make sure you have atleast python 3.9 installed, if for some reason "python3" does not work, try using "python" instead.*
### Installing the source code
<pre>
git clone git@github.com:RasmusRynell/Predicting-NHL.git
</pre>

### Create environment

Navigate into to project and create an environment
<pre>
cd Predicting-NHL
python3 -m venv env
</pre>
### Activate environment
On Windows:
<pre>source env\Scripts\activate.bat </pre>
On Unix/MacOS:
<pre>source env/bin/activate </pre>

### Install packages
<pre>python3 -m pip install -r requirements.txt</pre>

<br><br/>
## How to use *(Under construction)*
*Side comment: The application is currently accessed through a terminal, this terminal can then in later builds be replaced by a more traditional and easy to use UI.*

### Starting the application
To first start the application make sure you have followed the instructions under "Installation". When that is done simple navigate to the "app" folder and write the following:
<pre>python3 main.py</pre>
The application is then started, to then do certain things just enter in a command.

### Commands

#### General
* "help (h) *Prints all currently available commands*

* "exit" (e) *Exits the application*

#### Dev
* "eval" (ev) *under construction*

* "und" *Refreshes/Updates the local NHL database*

* "and" *Add nicknames to the database*

* "ubd" *Add "old" bets (from bookies but that's located on a local file) to database*

* "gen" *Generate a CSV file containing all information for a player going back to 2017/09/15*

<br><br/>
## Contributors
- [RasmusRynell](https://github.com/RasmusRynell)
- [Awarty](https://github.com/Awarty)

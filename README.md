# Predicting NHL stats (shots on goal)

## Overview
This project explores the idea of using different machine learning techniques to determine different stats in NHL games. Research and testing of different techniques has previously mainly been done in [this](https://github.com/RasmusRynell/sports_betting_test) project.

### Current features
- About 5000 predictions for “Shots on goal” in NHL games from different betting sites
- A way to add more "new" predictions by copy-pasting them from sites into text files
- Request data from NHL’s own database to populate your own internal database (in order to use less internet and have a faster running program)
- Produce a CSV file containing all data related to a prediction, this includes player statistics, players teams statistics, and enemy teams statistics for as many games back as one wants (typically no more than 5)

### Futures features
- Load and select features from CSV file to be used in different ML techniques (Data preprocessing)
- Create a pipeline for both using and testing and evaluating ML techniques
- Add more data (maybe xG for both player and teams)
- Create a better way of scraping the web for data
- Create a better way of scraping the web for previous bets and their outcomes
- Create a user interface either as an app or on the web

### Long term new features
- Create a more robust framework for different sports with different types of data
- Live predictions

<br><br/>
## Inner workings *(Under construction)*
### Data collection
In order to perform different ML techniques, data is needed, for now we use NHL's own (free and very detailed) database to gather all our data. We take this data and store it in our own [sqlite](https://www.sqlite.org/index.html) database to be used and updated when one wants/needs. The reasoning behind having our own database is quite simple, it’s a lot of data to ask for each time we want to do a prediction. This combined with the fact that we in the end want this process to be done once each day in the season the number of times we ask for a specific game in the database gets quickly out of control.

#### NHL's database
In order to use the NHL database we followed [this](https://gitlab.com/dword4/nhlapi/-/blob/master/stats-api.md) incredible documentation. Since we know we want all game data for every game x seasons back all we had to do was loop through each day for that period and request all games that occurred on that day. We then took that data and put it in our own database in order to be used later. To update the database is then to only request data for games that have not yet (according to our database) been played, by doing it this way we don’t have to keep requesting data that never changes.

<br><br/>
### Preprocessing

#### Feature selection

#### Feature extraction

#### Dimensionality reduction

#### Missing data removal / prediction

#### Transformation

#### Discretization


<br><br/>
### Machine learning

<br><br/>
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

* "pre" *Preprocess a csv file according to a configuration file*

<br><br/>
## Contributors
- [RasmusRynell](https://github.com/RasmusRynell)
- [Awarty](https://github.com/Awarty)

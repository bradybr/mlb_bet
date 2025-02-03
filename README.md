# MLB Gameday Picks

![image](https://github.com/bradybr/mlb_bet/blob/main/images/busch.jpg?)

Codified MLB betting methodology from Joe Peta's book _Trading Bases_.

The project is broken into two files: 01_futures.py and 02_gameday.py.  The 01_futures.py file is run once during the preseason to calculate prior year base rates compared to actual performance, to make updates for offseason roster changes, and ultimately make current year-end total win predictions.  The output of this script is the starting point for each game day, starting with opending day.  During the regular season, the 02_gameday.py script makes daily adjustments to this base rate file for starting lineups pitchers, and compares the calculated true odds of each game to the Fan Duel implied odds, registering a "bet" flag if the former is greater than the later.

## Getting Started
Several of the sites needed to pull current stats render their HTML dynamically through JS, so to get around this you'll need to download Chrome Driver and place the .exe file in your working directory to use the Selenium scripts. 

https://developer.chrome.com/docs/chromedriver/downloads


## Futures

```
os.chdir('')
```

```
# =============================================================================
# TIME DIMENSIONS
# =============================================================================

# Last 3 years
yrs = range(2022, 2025)
```

## Gameday


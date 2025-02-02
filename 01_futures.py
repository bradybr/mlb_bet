import os
import pandas as pd
import unicodedata
import re
import time
import statsapi as sapi
import pybaseball as pyb
import statsmodels.formula.api as smf
import requests
from bs4 import BeautifulSoup
from selenium import webdriver 


os.chdir('')



# =============================================================================
# TIME DIMENSIONS
# =============================================================================

# Last 3 years
yrs = range(2022, 2025); list(yrs)

# Previous (last completed) / Current (upcoming) Seasons
PY = max(yrs)
CY = max(yrs) + 1



# =============================================================================
# TEAM MASTER
# =============================================================================

# Team Info (Gameday)
######################################################

# Create Teams List
teams_ls = ['Diamondbacks',
            'Braves',
            'Orioles',
            'Red Sox',
            'Cubs',
            'White Sox',
            'Reds',
            'Guardians',
            'Rockies',          
            'Tigers',
            'Astros',
            'Royals',
            'Angels',
            'Dodgers',
            'Marlins',
            'Brewers',
            'Twins',
            'Mets',
            'Yankees',
            'Athletics',
            'Phillies',
            'Pirates',
            'Padres',
            'Giants',
            'Mariners',
            'Cardinals',
            'Rays',
            'Rangers',
            'Blue Jays',
            'Nationals']

# Lookup Team IDs (Gameday)
team_master = [sapi.lookup_team(team) for team in teams_ls]

# Flatten list
team_master = [x
               for xs in team_master
               for x in xs]

team_master = pd.DataFrame(team_master).rename({'id': 'teamID'}, axis = 1)

# Correct for Oakland A's moving to Sacramento in 2025
team_master.loc[team_master.teamCode == 'ath', 'teamCode'] = 'oak'


# Team ID (BREF)
######################################################

pyb_teams = pyb.team_ids(2021)
pyb_teams['teamCode'] = pyb_teams['teamIDretro'].str.lower()
pyb_teams.rename({'teamID': 'pyb_teamID'}, axis = 1, inplace = True)


# Fan Duel Cross Ref Teams
######################################################
fd_teams = pd.DataFrame({'teamID' : [121,158,143,144,110,108,145,116,113,120,117,147,118,142,119,138,146,134,135,137,139,141,140,112,133,114,109,115,136,111],
                         'FD_teamName' : ['New York Mets',
                                          'Milwaukee Brewers',
                                          'Philadelphia Phillies',
                                          'Atlanta Braves',
                                          'Baltimore Orioles',
                                          'Los Angeles Angels',
                                          'Chicago White Sox',
                                          'Detroit Tigers',
                                          'Cincinnati Reds',
                                          'Washington Nationals',
                                          'Houston Astros',
                                          'New York Yankees',
                                          'Kansas City Royals',
                                          'Minnesota Twins',
                                          'Los Angeles Dodgers',
                                          'St. Louis Cardinals',
                                          'Miami Marlins',
                                          'Pittsburgh Pirates',
                                          'San Diego Padres',
                                          'San Francisco Giants',
                                          'Tampa Bay Rays',
                                          'Toronto Blue Jays',
                                          'Texas Rangers',
                                          'Chicago Cubs',
                                          'Oakland Athletics',
                                          'Cleveland Guardians',
                                          'Arizona Diamondbacks',
                                          'Colorado Rockies',
                                          'Seattle Mariners',
                                          'Boston Red Sox']})


# Merge
######################################################

team_master = team_master.merge(pyb_teams[['lgID','pyb_teamID','teamIDfg','teamIDBR','teamIDretro','teamCode','franchID']],
                                how = 'left',
                                on = 'teamCode',
                                suffixes = ('', '_DROP')).filter(regex = '^(?!.*_DROP)')

team_master = team_master.merge(fd_teams, how = 'left', on = 'teamID')



# =============================================================================
# CLUSTER LUCK ADJUSTMENTS
# =============================================================================

# PY TEAM - BATTING
######################################################
PY_team_bat = pyb.team_batting(yrs[0], PY)
#PY_team_bat = pyb.team_batting(2008, 2010)
PY_team_bat.sort_values(by = ['Team','Season'], inplace = True)
PY_team_bat = PY_team_bat.reset_index(drop = True)


# PY TEAM - PITCHING
######################################################
PY_team_pitch = pyb.team_pitching(yrs[0], PY)
#PY_team_pitch = pyb.team_pitching(2008, 2010)
PY_team_pitch.sort_values(by = ['Team','Season'], inplace = True)
PY_team_pitch = PY_team_pitch.reset_index(drop = True)



# Runs Scored
######################################################

# Calculate Hits per Runs
CL_RS = PY_team_bat.copy()
CL_RS['HPR'] = CL_RS['H'] / CL_RS['R']
CL_RS = CL_RS.loc[:,['Season','teamIDfg','H','R','HPR','OBP','SLG','ISO']]


# Regress Hits per Run
HPR_mod = smf.ols('HPR ~ OBP + SLG + ISO', CL_RS).fit()


# Predict for CY
CL_RS = CL_RS.sort_values(by = 'Season').groupby('teamIDfg').tail(1).reset_index(drop = True)
CL_RS['HPR_pred'] = HPR_mod.predict(CL_RS.loc[:,['OBP','SLG','ISO']])
CL_RS['CL_RS_ADJ'] = round(CL_RS['R'] - (CL_RS['H'] / CL_RS['HPR_pred'])) * -1

CL_RS = CL_RS.merge(team_master[['teamIDfg','teamID']], how = 'left', on = 'teamIDfg')
CL_RS.rename({'R': 'RS'}, axis = 1, inplace = True)
CL_RS['CL_RS_ADJ'].sum()



# Runs Allowed
######################################################

CL_RA = []

vars_ls = ['R','AB','H','2B','3B','HR','SF','HBP','BB']

for team in pyb_teams.teamIDBR:
    #for yr in ['2008','2009','2010']:
    for yr in list(yrs):
        try:
            gms = pyb.team_game_logs(yr, team, "pitching")[vars_ls]
            gms = [team] + [yr] + list(gms.sum())
        except:
            new_team = pyb_teams.loc[pyb_teams['teamIDBR'] == team, 'franchID']
            gms = pyb.team_game_logs(yr, new_team.item(), "pitching")[vars_ls]
            gms = [team] + [yr] + list(gms.sum())
            
        CL_RA.append(gms)


# Add in Team
CL_RA = pd.DataFrame(CL_RA)
CL_RA.columns = ['teamIDBR'] + ['Season'] + vars_ls


# Calculate metrics
CL_RA['HAPRA'] = CL_RA['H'] / CL_RA['R']
CL_RA['OBP'] = (CL_RA['H'] + CL_RA['BB'] + CL_RA['HBP']) / (CL_RA['AB'] + CL_RA['BB'] + CL_RA['HBP'] + CL_RA['SF'])
CL_RA['SLG'] = ((CL_RA['H'] - (CL_RA['2B'] + CL_RA['3B'] + CL_RA['HR'])) + (2 * CL_RA['2B']) + (3 * CL_RA['3B']) + (4 * CL_RA['HR'])) / CL_RA['AB']
CL_RA['ISO'] = (CL_RA['2B'] + (2 * CL_RA['3B']) + (3 * CL_RA['HR'])) / CL_RA['AB']


# Regress
HAPRA_mod = smf.ols('HAPRA ~ OBP + SLG + ISO', CL_RA).fit()


# Predict for CY
CL_RA = CL_RA.sort_values('Season').groupby('teamIDBR').tail(1).reset_index(drop = True)
CL_RA['HAPRA_pred'] = HAPRA_mod.predict(CL_RA.loc[:,['OBP','SLG','ISO']])
CL_RA['CL_RA_ADJ'] = round(CL_RA['R'] - (CL_RA['H'] / CL_RA['HAPRA_pred'])) * -1

CL_RA = CL_RA.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
CL_RA.rename({'R': 'RA'}, axis = 1, inplace = True)
CL_RA['CL_RA_ADJ'].sum()


# Compare RS/RA
######################################################
CL_RS['CL_RS_ADJ'].sum()
CL_RA['CL_RA_ADJ'].sum()



# =============================================================================
# WAR ADJUSTMENTS
# =============================================================================

# Previous Year
######################################################

# Batting
###############################
# URL
url = f"https://www.fangraphs.com/leaders/major-league?pos=all&stats=bat&lg=all&qual=0&type=8&season={PY}&season1={PY}&ind=0&team=0%2Cts&rost=&age=&filter=&players=0&pageitems=30&month=0"

# Out
PY_WAR_bat = []

# Selenium freeze
driver = webdriver.Chrome(os.getcwd() + '\\chromedriver.exe')
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html)
driver.quit()

for item in soup.select('div.table-scroll > table > tbody > tr'):
    team = item.find('td', {'data-stat': 'Team'}).text
    WAR = item.find('td', {'data-stat': 'WAR'}).text
    
    PY_WAR_bat.append([team, PY, WAR])


# Create DF
PY_WAR_bat = pd.DataFrame(PY_WAR_bat).rename({0: 'teamIDBR',
                                              1: 'season',
                                              2: 'PY_WAR_bat'}, axis = 1)

PY_WAR_bat['PY_WAR_bat'] = PY_WAR_bat['PY_WAR_bat'].astype('float')



# SP Pitching
###############################
url = f"https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=0&type=8&season={PY}&season1={PY}&ind=0&team=0%2Cts&rost=&age=&filter=&players=0&month=0&pageitems=30&stats=sta"

PY_WAR_SP = []

# Selenium freeze
driver = webdriver.Chrome(os.getcwd() + '\\chromedriver.exe')
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html)
driver.quit()

for item in soup.select('div.table-scroll > table > tbody > tr'):
    team = item.find('td', {'data-stat': 'Team'}).text
    WAR = item.find('td', {'data-stat': 'WAR'}).text
    
    PY_WAR_SP.append([team, PY, WAR])

# Create DF
PY_WAR_SP = pd.DataFrame(PY_WAR_SP).rename({0: 'teamIDBR',
                                            1: 'season',
                                            2: 'PY_WAR_SP'}, axis = 1)

PY_WAR_SP['PY_WAR_SP'] = PY_WAR_SP['PY_WAR_SP'].astype('float')



# RP Pitching
###############################
url = f"https://www.fangraphs.com/leaders/major-league?pos=all&lg=all&qual=0&type=8&season={PY}&season1={PY}&ind=0&team=0%2Cts&rost=&age=&filter=&players=0&month=0&pageitems=30&stats=rel"

PY_WAR_RP = []

# Selenium freeze
driver = webdriver.Chrome(os.getcwd() + '\\chromedriver.exe')
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html)
driver.quit()

for item in soup.select('div.table-scroll > table > tbody > tr'):
    team = item.find('td', {'data-stat': 'Team'}).text
    WAR = item.find('td', {'data-stat': 'WAR'}).text
    
    PY_WAR_RP.append([team, PY, WAR])

# Create DF
PY_WAR_RP = pd.DataFrame(PY_WAR_RP).rename({0: 'teamIDBR',
                                            1: 'season',
                                            2: 'PY_WAR_RP'}, axis = 1)

# Calculate League Avg
PY_WAR_RP = PY_WAR_RP.merge(team_master[['teamIDBR','lgID']], how = 'left', on = 'teamIDBR')
PY_WAR_RP['PY_WAR_RP'] = PY_WAR_RP['PY_WAR_RP'].astype('float')

PY_LGWAR_RP = PY_WAR_RP.groupby('lgID')['PY_WAR_RP'].mean().reset_index()
PY_LGWAR_RP.rename({'PY_WAR_RP': 'CY_LGWAR_RP'}, axis = 1, inplace = True)

PY_WAR_RP = PY_WAR_RP.merge(PY_LGWAR_RP, how = 'left', on = 'lgID')



# Current Year
######################################################

url = 'https://www.fangraphs.com/depthcharts.aspx?position=Team'


# Batting
###############################
CY_WAR_bat = []

# Selenium freeze
driver = webdriver.Chrome(os.getcwd() + '\\chromedriver.exe')
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html)
driver.quit()

for item in soup.select('div.depth-charts-aspx_table > table > tbody > tr'):
    team = item.select('a')[0].text
    WAR = item.select('td')[12].text
    
    CY_WAR_bat.append([team, CY, WAR])


# Create DF
CY_WAR_bat = pd.DataFrame(CY_WAR_bat).rename({0: 'teamName',
                                              1: 'Season',
                                              2: 'CY_WAR_bat'}, axis = 1)

CY_WAR_bat['CY_WAR_bat'] = CY_WAR_bat['CY_WAR_bat'].astype('float')

CY_WAR_bat.loc[CY_WAR_bat['teamName'] == 'Diamondbacks', 'teamName'] = 'D-backs'

CY_WAR_bat = CY_WAR_bat.merge(team_master[['teamName','teamIDBR']], how = 'left', on = 'teamName')



# Pitching
###############################
CY_WAR_pitch = []

for item in soup.select('div.depth-charts-aspx_table > table > tbody > tr'):
    team = item.select('a')[0].text
    SP_WAR = item.select('td')[10].text
    RP_WAR = item.select('td')[11].text
    
    CY_WAR_pitch.append([team, CY, SP_WAR, RP_WAR])


# Create DFs
CY_WAR_pitch = pd.DataFrame(CY_WAR_pitch).rename({0: 'teamName',
                                                  1: 'Season',
                                                  2: 'CY_WAR_SP',
                                                  3: 'CY_WAR_RP'}, axis = 1) 

CY_WAR_pitch['CY_WAR_SP'] = CY_WAR_pitch['CY_WAR_SP'].astype('float')
CY_WAR_pitch['CY_WAR_RP'] = CY_WAR_pitch['CY_WAR_RP'].astype('float')

CY_WAR_pitch.loc[CY_WAR_pitch['teamName'] == 'Diamondbacks', 'teamName'] = 'D-backs'

CY_WAR_pitch = CY_WAR_pitch.merge(team_master[['teamName','teamIDBR']], how = 'left', on = 'teamName')



# =============================================================================
# TEAM STATS
# =============================================================================

# Previous Year
PY_stats = sapi.standings_data(season = PY)
PY_WL = []

for item in PY_stats:
    for team in PY_stats[item]['teams']:
        PY_WL.append(team)

PY_WL = pd.DataFrame(PY_WL).rename({'team_id': 'teamID',
                                    'w': 'PY_W',
                                    'l': 'PY_L'}, axis = 1)

PY_WL['PY_W_PCT'] = PY_WL.PY_W / (PY_WL.PY_W + PY_WL.PY_L)
PY_WL = PY_WL.merge(team_master[['teamID','teamCode']], how = 'left', on = 'teamID')



# =============================================================================
# CY PYTH
# =============================================================================

print(list(team_master.columns))
print(list(PY_WL.columns))
print(list(CL_RS.columns))
print(list(CL_RA.columns))
print(list(PY_WAR_bat.columns))
print(list(PY_WAR_SP.columns))
print(list(PY_WAR_RP.columns))
print(list(CY_WAR_bat.columns))
print(list(CY_WAR_pitch.columns))


dat = PY_WL[['teamID','PY_W','PY_W_PCT']].copy()
dat['PY_W_PCT'] = round(dat['PY_W_PCT'], 3)
dat = dat.merge(team_master[['teamID','teamIDBR','teamName']], how = 'left', on = 'teamID')
dat = dat.merge(CL_RS[['teamID','RS','CL_RS_ADJ']], how = 'left', on = 'teamID')
dat = dat.merge(CL_RA[['teamID','RA','CL_RA_ADJ']], how = 'left', on = 'teamID')
dat['PY_PYT_W_PCT'] = round((dat['RS'] ** 1.83) / ((dat['RS'] ** 1.83) + (dat['RA'] ** 1.83)), 3)
dat['PY_PYT_W'] = round(dat['PY_PYT_W_PCT'] * 162)
dat['PY_PYT_W_DIFF'] = dat['PY_W'] - dat['PY_PYT_W']

dat['PY_RS_NEW'] = dat['RS'] + dat['CL_RS_ADJ']
dat['PY_RA_NEW'] = dat['RA'] + dat['CL_RA_ADJ']
dat['PY_RS_CHG'] = dat['PY_RS_NEW'] - dat['RS']
dat['PY_RA_CHG'] = dat['PY_RA_NEW'] - dat['RA']


WAR = PY_WAR_bat[['teamIDBR','PY_WAR_bat']].copy()
WAR = WAR.merge(PY_WAR_SP[['teamIDBR','PY_WAR_SP']], how = 'left', on = 'teamIDBR')
WAR = WAR.merge(PY_WAR_RP[['teamIDBR','PY_WAR_RP','CY_LGWAR_RP']], how = 'left', on = 'teamIDBR')
WAR = WAR.merge(CY_WAR_bat[['teamIDBR','CY_WAR_bat']], how = 'left', on = 'teamIDBR')
WAR = WAR.merge(CY_WAR_pitch[['teamIDBR','CY_WAR_SP']], how = 'left', on = 'teamIDBR')


WAR['BAT_RS_ADJ'] = round((WAR.CY_WAR_bat - WAR.PY_WAR_bat) * 10)
WAR['SP_RA_ADJ'] = round((WAR.PY_WAR_SP - WAR.CY_WAR_SP) * 10)
WAR['RP_RA_ADJ'] = round((WAR.PY_WAR_RP - WAR.CY_LGWAR_RP) * 10)

dat = dat.merge(WAR[['teamIDBR','CY_WAR_bat','CY_WAR_SP','CY_LGWAR_RP','BAT_RS_ADJ','SP_RA_ADJ','RP_RA_ADJ']], on = 'teamIDBR')

dat['CY_RS_NEW'] = dat['RS'] + dat['CL_RS_ADJ'] + dat['BAT_RS_ADJ']
dat['CY_RA_NEW'] = dat['RA'] + dat['CL_RA_ADJ'] + dat['SP_RA_ADJ'] + dat['RP_RA_ADJ']


# Normalize to total RS/RA environment
dat['CY_RS_NEW'].sum()
dat['CY_RA_NEW'].sum()

if dat['CY_RS_NEW'].sum() > dat['CY_RA_NEW'].sum():
    norm_var = 'CY_RS_NEW'
    chg_var = 'CY_RA_NEW'
else:
    norm_var = 'CY_RA_NEW'
    chg_var = 'CY_RS_NEW'

dat['norm_PCT'] = dat[chg_var] / dat[chg_var].sum()
dat[chg_var] = round(dat['norm_PCT'] * dat[norm_var].sum())

dat['CY_RS_NEW'].sum()
dat['CY_RA_NEW'].sum()


# CY PYTH Projections
dat['CY_PYT_W_PCT'] = round((dat['CY_RS_NEW'] ** 1.83) / ((dat['CY_RS_NEW'] ** 1.83) + (dat['CY_RA_NEW'] ** 1.83)), 3)
dat['CY_W'] = round(dat['CY_PYT_W_PCT'] * 162)
dat['CY_W_CHG'] = dat['CY_W'] - dat['PY_W']


list(dat.columns)
dat = dat[['teamName',
         'teamIDBR',
         'teamID',
         'RS',
         'RA',
         'PY_W',
         'PY_W_PCT',
         'PY_PYT_W',         
         'PY_PYT_W_DIFF',
         'CL_RS_ADJ',
         'CL_RA_ADJ',
         #'PY_RS_NEW',
         #'PY_RS_CHG',
         #'PY_RA_NEW',
         #'PY_RA_CHG',
         #'CY_WAR_bat',
         #'CY_WAR_SP',
         #'CY_WAR_RP',
         'BAT_RS_ADJ',
         'SP_RA_ADJ',
         'RP_RA_ADJ',
         'CY_RS_NEW',
         'CY_RA_NEW',
         'CY_W',
         'CY_W_CHG',
         'CY_PYT_W_PCT']]



# =============================================================================
# EXPORTS
# =============================================================================

# Opening Day Win Expectancy
dat.to_csv("OD_W_EXP.csv", index = False)

# Team Master
team_master.to_csv("team_master.csv", index = False)

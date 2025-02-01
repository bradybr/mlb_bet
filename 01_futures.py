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


os.chdir('D:/Projects/MLB2')



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
driver = webdriver.Chrome("D:/Projects/MLB2/chromedriver.exe")
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
driver = webdriver.Chrome("D:/Projects/MLB2/chromedriver.exe")
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
driver = webdriver.Chrome("D:/Projects/MLB2/chromedriver.exe")
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
driver = webdriver.Chrome("D:/Projects/MLB2/chromedriver.exe")
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





























# =============================================================================
# PLAYER MASTER
# =============================================================================

# Previous Year - Fangraphs
######################################################

PY_players_fg = pyb.batting_stats(PY, ind = 1, qual = 0)

# Lookup Other Ids
player_ids = pyb.playerid_reverse_lookup(PY_players_fg.IDfg, key_type = 'fangraphs')

# Merge
PY_players_fg = PY_players_fg.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', left_on = 'IDfg', right_on = 'key_fangraphs')
PY_players_fg['key_fangraphs'] = PY_players_fg['IDfg']
PY_players_fg.loc[pd.isnull(PY_players_fg.key_fangraphs), 'key_fangraphs'] = PY_players_fg.IDfg


# Previous Year - Gameday
######################################################

players_master = []

for id_ in team_master.teamID:
    
    # Clean up environment
    del_ls = ['team','pos','player','player_id']
    
    for obj_nm in del_ls:
        if obj_nm in globals():
            globals().pop(obj_nm)

    # Query 40 Man Rosters
    team_roster = sapi.get("team_roster", {"rosterType": '40man', "season": PY, "teamId": id_, "hydrate": "person"})

    # Extract details
    try:
        for player in range(0, len(team_roster['roster'])):
            pos =       team_roster['roster'][player]['position']['abbreviation']
            team =      team_roster['teamId']
            player_id = team_roster['roster'][player]['person']['id']
            player =    team_roster['roster'][player]['person']['fullName']            
    
            players_master.append([team, pos, player, player_id])

    except:
        pass


# Create DF
players_master = pd.DataFrame(players_master).rename({0: 'teamID',
                                                      1: 'pos',
                                                      2: 'name',        
                                                      3: 'key_mlbam'}, axis = 1)

players_master['teamID'].nunique()


# Lookup Other Ids
player_ids = pyb.playerid_reverse_lookup(players_master.key_mlbam, key_type = 'mlbam')

# Merge
players_master = players_master.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_mlbam')


# Remove Accents (~) function
def remove_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

# Remove accents (~) from names
players_master.name = [remove_accents(x) for x in players_master.name]


# Merge & impute for any missing fangraph IDs
players_master = players_master.merge(PY_players_fg[['Name','IDfg']], how = 'left', left_on = 'name', right_on = 'Name')


players_master.loc[players_master['key_fangraphs'] == -1, 'key_fangraphs'] = players_master['IDfg']
players_master.drop('IDfg', axis = 1, inplace = True)
players_master.drop_duplicates(inplace = True)

































# =============================================================================
# TEAM ROSTERS
# =============================================================================

# PY STARTING LINEUPS
######################################################

# URL containers
pre_url = 'https://www.baseball-reference.com/teams/'
url_team =[]
for team in pyb_teams.teamIDBR:
    url_team.append(team)

PY = str(yrs[len(yrs)-1])
post_url = '.shtml'


# Build URL strings
urls = []
for team in url_team:
    urls.append(pre_url + team + "/" + PY + post_url)

# Out
PY_rosters_bat = []
PY_rosters_pitch = []


# Parse
for url in urls:
    
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')    

    # Batting
    for item in soup.select('div#div_players_standard_batting > table > tbody > tr'):
        
        # Clean up environment
        del_ls = ['team','rank','pos','player','player_id']
        
        for obj_nm in del_ls:
            if obj_nm in globals():
                globals().pop(obj_nm)
        
        try:
            team = url.split('/')[4]
            rank = item.select('th')[0].text
            pos = item.select('td')[2].text
            player = re.sub('[^A-Za-z ]+', '', item.select('td')[0].text)
            player_id = item.select('td')[0]['data-append-csv']
            games = item.find('td', {'data-stat': 'b_games'}).text
            at_bats = item.find('td', {'data-stat': 'b_ab'}).text
              
            # Out
            PY_rosters_bat.append([team, rank, pos, player, player_id, games, at_bats])
    
        except:
            pass
    
    # Pitching
    #page = BeautifulSoup(requests.get(url).content, "lxml")
    #soup = BeautifulSoup("\n".join(page.find_all(string = Comment)), "lxml")
    
    for item in soup.select('div#div_players_standard_pitching > table > tbody > tr'): 
    
        # Clean up environment
        del_ls = ['team','rank','pos','player','player_id']
        
        for obj_nm in del_ls:
            if obj_nm in globals():
                globals().pop(obj_nm)
            
        try:
            team = url.split('/')[4]
            rank = item.select('th')[0].text
            pos = item.select('td')[2].text
            player = re.sub('[^A-Za-z ]+', '',item.select('td')[0].text)
            player_id = item.select('td')[0]['data-append-csv']
            games = item.find('td', {'data-stat': 'p_g'}).text
            ins_pitched = item.find('td', {'data-stat': 'p_ip'}).text
            
            # Out
            PY_rosters_pitch.append([team, rank, pos, player, player_id, games, ins_pitched])
    
        except:
            pass
    
    time.sleep(2)


# Create DFs
PY_rosters_pitch = pd.DataFrame(PY_rosters_pitch).rename({0: 'teamIDBR',
                                                          1: 'rank',
                                                          2: 'pos',
                                                          3: 'name',
                                                          4: 'key_bbref',
                                                          5: 'G',
                                                          6: 'AB'}, axis = 1)

PY_rosters_bat = pd.DataFrame(PY_rosters_bat).rename({0: 'teamIDBR',
                                                      1: 'rank',
                                                      2: 'pos',
                                                      3: 'name',
                                                      4: 'key_bbref',
                                                      5: 'G',
                                                      6: 'IP'}, axis = 1)

PY_rosters_bat['teamIDBR'].nunique()
PY_rosters_pitch['teamIDBR'].nunique()


# Lookup Fangraphs ID
PY_rosters_bat = PY_rosters_bat.merge(players_master[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_bbref')
player_ids = pyb.playerid_reverse_lookup(PY_rosters_bat.key_bbref, key_type = 'bbref')

# Merge
a = PY_rosters_bat.copy()
a = a.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_bbref')





PY_rosters_pitch = PY_rosters_pitch.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_bbref')




















# Batting
PY_rosters_bat[['rank']] = PY_rosters_bat[['rank']].astype('int')

PY_players_bat = PY_rosters_bat.query('rank <= 9').sort_values(by = ['teamIDBR','rank'])
PY_players_bat = PY_rosters_bat.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
PY_players_bat.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_bat'}, axis = 1)
PY_players_bat.groupby('teamIDBR').size()


# Pitching (Starting & Relief)
PY_rosters_pitch[['rank']] = PY_rosters_pitch[['rank']].astype('int')
PY_rosters_pitch = PY_rosters_pitch.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
PY_rosters_pitch.groupby('teamIDBR').size()

PY_players_SP = PY_rosters_pitch[(~PY_rosters_pitch['pos'].isin(['CL', 'RP'])) & (PY_rosters_pitch.GS > 0)]
PY_players_SP = PY_players_SP.sort_values(by = ['teamIDBR','rank']).groupby('teamIDBR').head(5)
PY_players_SP.groupby('teamIDBR').size()

PY_players_RP = PY_rosters_pitch[(~PY_rosters_pitch['pos'].isin(['SP'])) & (PY_rosters_pitch.GS == 0)]
PY_players_RP = PY_players_RP.sort_values(by = ['teamIDBR','rank']).groupby('teamIDBR').head(5)
PY_players_RP.groupby('teamIDBR').size()


# WAR (Starting Lineups)
PY_players_bat.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_bat'}, axis = 1)
PY_players_SP.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)
PY_players_RP.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)


# League RP WAR
LG_RP_WAR = PY_players_RP[PY_players_RP.pos != ""].groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)
LG_RP_WAR = LG_RP_WAR.merge(team_master[['teamIDBR','lgID']], how = 'left', on = 'teamIDBR', suffixes = ('', '_DROP')).filter(regex = '^(?!.*_DROP)')
LG_RP_WAR = LG_RP_WAR.groupby('lgID')['WAR_pitch'].mean().reset_index()
              






















# =============================================================================
# TEAM ROSTERS
# =============================================================================

# Previous Year - All
######################################################

# URL containers
pre_url = 'https://www.baseball-reference.com/teams/'
url_team =[]
for team in pyb_teams.teamIDBR:
    url_team.append(team)

PY = str(yrs[len(yrs)-1])
post_url = '.shtml'


# Build URL strings
urls = []
for team in url_team:
    urls.append(pre_url + team + "/" + PY + post_url)

# Out
PY_rosters_bat = []
PY_rosters_pitch = []


# Parse
for url in urls:
    
    # Selenium freeze
    driver = webdriver.Chrome("D:/Projects/MLB2/chromedriver.exe")
    driver.get(url)
    html = driver.page_source
    soup = BeautifulSoup(html)
    driver.quit()
    
    # Batting
    for item in soup.select('div#all_players_standard_batting > div > table > tbody > tr'):
        
        # Clean up environment
        del_ls = ['team','rank','pos','player','player_id']
        
        for obj_nm in del_ls:
            if obj_nm in globals():
                globals().pop(obj_nm)
        
        try:
            team = url.split('/')[4]
            rank = item.select('th')[0].text
            pos = item.select('td')[2].text
            player = re.sub('[^A-Za-z ]+', '', item.select('td')[0].text)
            player_id = item.select('td')[0]['data-append-csv']
              
            # Out
            PY_rosters_bat.append([team, rank, pos, player, player_id])
    
        except:
            pass
    
    # Pitching
    #page = BeautifulSoup(requests.get(url).content, "lxml")
    #soup = BeautifulSoup("\n".join(page.find_all(string = Comment)), "lxml")
    
    for item in soup.select('div#div_players_standard_pitching > table > tbody > tr'): 
    #className = "table_container tabbed current is_setup"
    #for item in soup.find_all("div", {"class": className.split() if ' ' in className else className})[0].find_all('tbody')[0].find_all('tr'):
    
        # Clean up environment
        del_ls = ['team','rank','pos','player','player_id']
        
        for obj_nm in del_ls:
            if obj_nm in globals():
                globals().pop(obj_nm)
            
        try:
            team = url.split('/')[4]
            rank = item.select('th')[0].text
            pos = item.select('td')[2].text
            player = re.sub('[^A-Za-z ]+', '',item.select('td')[0].text)
            player_id = item.select('td')[0]['data-append-csv']
            
            # Out
            PY_rosters_pitch.append([team, rank, pos, player, player_id])
    
        except:
            pass
    
    time.sleep(2)


# Create DFs
PY_rosters_pitch = pd.DataFrame(PY_rosters_pitch).rename({0: 'teamIDBR',
                                                          1: 'rank',
                                                          2: 'pos',
                                                          3: 'name',
                                                          4: 'key_bbref'}, axis = 1)

PY_rosters_bat = pd.DataFrame(PY_rosters_bat).rename({0: 'teamIDBR',
                                                      1: 'rank',
                                                      2: 'pos',
                                                      3: 'name',
                                                      4: 'key_bbref'}, axis = 1)

PY_rosters_bat['teamIDBR'].nunique()
PY_rosters_pitch['teamIDBR'].nunique()


# Lookup Fangraphs ID
player_ids = pyb.playerid_reverse_lookup(pd.concat([PY_rosters_pitch.key_bbref, PY_rosters_bat.key_bbref]), key_type = 'bbref')


# Merge
PY_rosters_bat = PY_rosters_bat.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_bbref')
PY_rosters_pitch = PY_rosters_pitch.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_bbref')







# PY WAR (40 man Roster)
######################################################

# BATTING (Fangraphs)
PY_WAR_bat = pyb.batting_stats(yrs[len(yrs)-1], ind = 1, qual = 1)




PY_WAR_bat = PY_WAR_bat.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', left_on = 'IDfg', right_on = 'key_fangraphs')

# Combine PY Rosters with WAR
PY_rosters_bat = PY_rosters_bat.merge(PY_WAR_bat[['key_mlbam', 'WAR']], how = 'left', on = 'key_mlbam')


# PITCHING (Fangraphs)
PY_WAR_pitch = pyb.pitching_stats(yrs[len(yrs)-1], ind = 1, qual = 1)[['IDfg','Season','Name','G','GS','IP','Start-IP','Relief-IP','xFIP','SIERA','WAR']]
PY_WAR_pitch = PY_WAR_pitch.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', left_on = 'IDfg', right_on = 'key_fangraphs')

# Combine PY Rosters with WAR
PY_rosters_pitch = PY_rosters_pitch.merge(PY_WAR_pitch[['key_mlbam','Season','Name','G','GS','IP','Start-IP','Relief-IP','xFIP','SIERA','WAR']], how = 'left', on = 'key_mlbam')



# PY STARTING LINEUPS
######################################################

# Batting
PY_rosters_bat[['rank']] = PY_rosters_bat[['rank']].astype('int')

PY_players_bat = PY_rosters_bat.query('rank <= 9').sort_values(by = ['teamIDBR','rank'])
PY_players_bat = PY_rosters_bat.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
PY_players_bat.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_bat'}, axis = 1)
PY_players_bat.groupby('teamIDBR').size()


# Pitching (Starting & Relief)
PY_rosters_pitch[['rank']] = PY_rosters_pitch[['rank']].astype('int')
PY_rosters_pitch = PY_rosters_pitch.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
PY_rosters_pitch.groupby('teamIDBR').size()

PY_players_SP = PY_rosters_pitch[(~PY_rosters_pitch['pos'].isin(['CL', 'RP'])) & (PY_rosters_pitch.GS > 0)]
PY_players_SP = PY_players_SP.sort_values(by = ['teamIDBR','rank']).groupby('teamIDBR').head(5)
PY_players_SP.groupby('teamIDBR').size()

PY_players_RP = PY_rosters_pitch[(~PY_rosters_pitch['pos'].isin(['SP'])) & (PY_rosters_pitch.GS == 0)]
PY_players_RP = PY_players_RP.sort_values(by = ['teamIDBR','rank']).groupby('teamIDBR').head(5)
PY_players_RP.groupby('teamIDBR').size()


# WAR (Starting Lineups)
PY_players_bat.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_bat'}, axis = 1)
PY_players_SP.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)
PY_players_RP.groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)


# League RP WAR
LG_RP_WAR = PY_players_RP[PY_players_RP.pos != ""].groupby('teamIDBR')['WAR'].sum().reset_index().rename({'WAR': 'WAR_pitch'}, axis = 1)
LG_RP_WAR = LG_RP_WAR.merge(team_master[['teamIDBR','lgID']], how = 'left', on = 'teamIDBR', suffixes = ('', '_DROP')).filter(regex = '^(?!.*_DROP)')
LG_RP_WAR = LG_RP_WAR.groupby('lgID')['WAR_pitch'].mean().reset_index()
              


# PY SP SIERA
######################################################

PY_players_SP.to_csv("PY_players_SP.csv", index = False)




# =============================================================================
# CLUSTER LUCK ADJUSTMENTS
# =============================================================================

# PY TEAM - BATTING
######################################################
PY_team_bat = pyb.team_batting(yrs[0], yrs[len(yrs)-1])
PY_team_bat.sort_values(by = ['Team','Season'], inplace = True)
PY_team_bat = PY_team_bat.reset_index(drop = True)
#PY_team_bat.rename({'Team': 'franchID'}, axis = 1, inplace = True)



# PY TEAM - PITCHING
######################################################
PY_team_pitch = pyb.team_pitching(yrs[0], yrs[len(yrs)-1])
PY_team_pitch.sort_values(by = ['Team','Season'], inplace = True)
PY_team_pitch = PY_team_pitch.reset_index(drop = True)
#PY_team_pitch.rename({'Team': 'franchID'}, axis = 1, inplace = True)



# Runs Scored
######################################################

# Calculate Hits per Runs
CL_RS = PY_team_bat.copy()
CL_RS['HPR'] = CL_RS['H'] / CL_RS['R']
CL_RS = CL_RS.loc[:,['Season','teamIDfg','H','R','HPR','OBP','SLG','ISO']]


# Regress Hits per Run
HPR_mod = smf.ols('HPR ~ OBP + SLG + ISO', CL_RS).fit()


# Predict for CY
CL_RS = CL_RS.groupby('teamIDfg').tail(1).reset_index(drop = True)
CL_RS['HPR_pred'] = HPR_mod.predict(CL_RS.loc[:,['OBP','SLG','ISO']])
CL_RS['CL_RS_ADJ'] = round(CL_RS['R'] - (CL_RS['H'] / CL_RS['HPR_pred'])) * -1

CL_RS = CL_RS.merge(team_master[['teamIDfg','teamID']], how = 'left', on = 'teamIDfg')
CL_RS.rename({'R': 'RS'}, axis = 1, inplace = True)
CL_RS['CL_RS_ADJ'].sum()



# Runs Allowed
######################################################

#pyb_teams = pyb.team_ids(yrs[len(yrs)-1])

CL_RA = []

vars_ls = ['R','AB','H','2B','3B','HR','SF','HBP','BB']

for team in pyb_teams.teamIDBR:
    try:
        gms = pyb.team_game_logs(yrs[len(yrs)-1], team, "pitching")[vars_ls]
        gms = [team] + list(gms.sum())
    except:
        new_team = pyb_teams.loc[pyb_teams['teamIDBR'] == team, 'franchID']
        gms = pyb.team_game_logs(yrs[len(yrs)-1], new_team.item(), "pitching")[vars_ls]
        gms = [team] + list(gms.sum())
        
    CL_RA.append(gms)


# Add in Team
CL_RA = pd.DataFrame(CL_RA)
CL_RA.columns = ['teamIDBR'] + vars_ls


# Calculate metrics
CL_RA['HAPRA'] = CL_RA['H'] / CL_RA['R']
CL_RA['OBP'] = (CL_RA['H'] + CL_RA['BB'] + CL_RA['HBP']) / (CL_RA['AB'] + CL_RA['BB'] + CL_RA['HBP'] + CL_RA['SF'])
CL_RA['SLG'] = ((CL_RA['H'] - (CL_RA['2B'] + CL_RA['3B'] + CL_RA['HR'])) + (2 * CL_RA['2B']) + (3 * CL_RA['3B']) + (4 * CL_RA['HR'])) / CL_RA['AB']
CL_RA['ISO'] = (CL_RA['2B'] + (2 * CL_RA['3B']) + (3 * CL_RA['HR'])) / CL_RA['AB']


# Regress
HAPRA_mod = smf.ols('HAPRA ~ OBP + SLG + ISO', CL_RA).fit()


# Predict for CY
CL_RA['HAPRA_pred'] = HAPRA_mod.predict(CL_RA.loc[:,['OBP','SLG','ISO']])
CL_RA['CL_RA_ADJ'] = round(CL_RA['R'] - (CL_RA['H'] / CL_RA['HAPRA_pred'])) * -1

CL_RA = CL_RA.merge(team_master[['teamIDBR','teamID']], how = 'left', on = 'teamIDBR')
CL_RA.rename({'R': 'RA'}, axis = 1, inplace = True)
CL_RA['CL_RA_ADJ'].sum()




# =============================================================================
# ROSTER CHANGES ADJUSTMENTS
# =============================================================================

# CY Rosters (40 man)
######################################################

CY = max(yrs) + 1

CY_rosters = []

for id_ in team_master.teamID:
    team_roster = sapi.get("team_roster", {"rosterType": '40man', "season": CY, "teamId": id_, "hydrate": "person"})
    
    if 'team' in globals():
        del team
    if 'pos' in globals():
        del pos
    if 'player' in globals():
        del player
    if 'player_id' in globals():
        del player_id
            
    try:
        # Do not re-order (only works in this order for some reason)
        for player in range(0, len(team_roster['roster'])):
            pos =       team_roster['roster'][player]['position']['abbreviation']
            team =      team_roster['teamId']
            player_id = team_roster['roster'][player]['person']['id']
            player =    team_roster['roster'][player]['person']['fullName']            
    
            CY_rosters.append([team, pos, player, player_id])

    except:
        pass



# Create DF
CY_rosters = pd.DataFrame(CY_rosters).rename({0: 'teamID',
                                              1: 'pos',
                                              2: 'name',
                                              3: 'key_mlbam'}, axis = 1)

CY_rosters['teamID'].nunique()


# Lookup Fangraphs ID
player_ids = pyb.playerid_reverse_lookup(CY_rosters.key_mlbam, key_type = 'mlbam')

# Merge
CY_rosters = CY_rosters.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_mlbam')




# WAR
######################################################

# URL containers (Fangraphs)
pre_url = 'https://www.fangraphs.com/depthcharts.aspx?position=ALL&teamid='

urls =[]
for team in range(1, len(pyb_teams.teamIDBR) + 1):
    urls.append(pre_url + str(team))


# Out
CY_WAR_bat = []
CY_WAR_pitch = []


for url in urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.text, 'html.parser')
    
    className = "tablesoreder, depth_chart tablesorter tablesorter-default"
    team = re.search('(?<=-)(.*)(?=\|)', str(soup.select('head > title')))[0].strip()
    
    for item in soup.find_all('table', {'class': className.split() if ' ' in className else className})[9]:

        
        if 'player' in globals():
            del player
        if 'player_id' in globals():
            del player_id
        if 'WAR' in globals():
            del WAR
        try:
            player = item.select('a')[0].text
            player_id = int(''.join(re.sub('[^0-9]+', '', re.search('(?<=playerid=)(.*)(?=&)', item.select('a')[0]['href'])[0])))
            pos = ''.join(re.split('/', re.findall('(?<=position=)(.*)', item.select('a')[0]['href'])[0])[0])
            WAR =  item.select('td')[9].text
           
            # Out
            CY_WAR_bat.append([team, player, player_id, pos, WAR])

        except:
            pass
        
    for item in soup.find_all('table', {'class': className.split() if ' ' in className else className})[12]:
        if 'player' in globals():
            del player
        if 'player_id' in globals():
            del player_id
        if 'WAR' in globals():
            del WAR
            
        try:
            player = item.select('a')[0].text
            player_id = int(''.join(re.sub('[^0-9]+', '', re.search('(?<=playerid=)(.*)(?=&)', item.select('a')[0]['href'])[0])))
            pos = ''.join(re.split('/', re.findall('(?<=position=)(.*)', item.select('a')[0]['href'])[0])[0])
            WAR =  item.select('td')[9].text
               
            # Out
            CY_WAR_pitch.append([team, player, player_id, pos, WAR])

        except:
            pass
    
    time.sleep(2)
    

# Create DFs
CY_WAR_bat = pd.DataFrame(CY_WAR_bat).rename({0: 'teamName',
                                              1: 'player',
                                              2: 'key_fangraphs',
                                              3: 'pos',
                                              4: 'WAR'}, axis = 1)

CY_WAR_pitch = pd.DataFrame(CY_WAR_pitch).rename({0: 'teamName',
                                                  1: 'player',
                                                  2: 'key_fangraphs',
                                                  3: 'pos',
                                                  4: 'WAR'}, axis = 1)


CY_WAR = pd.concat([CY_WAR_bat, CY_WAR_pitch])


# Lookup Fangraphs ID
player_ids = pyb.playerid_reverse_lookup(CY_WAR.key_fangraphs, key_type = 'fangraphs')
CY_WAR = CY_WAR.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_fangraphs')

CY_WAR['teamName'].sort_values().unique()
team_master['teamName'].sort_values().unique()

CY_WAR.loc[CY_WAR['teamName'] == 'Diamondbacks', 'teamName'] = 'D-backs'

CY_WAR = CY_WAR.merge(team_master[['teamName','teamID']], how = 'left', on = 'teamName')
CY_WAR

CY_WAR.to_csv("CY_WAR.csv", index = False)


# Add WAR into CY Rosters
CY_rosters = CY_rosters.merge(CY_WAR[['key_mlbam', 'WAR']], how = 'outer', on = 'key_mlbam')




# =============================================================================
# LIVE STARTING LINEUP ADJUSTMENTS
# =============================================================================

# READ-IN 3/11
OD_rosters = pd.read_csv('OD_rosters.csv')



################## NOT RUN ####################
# Requires manual fill ins below if you run this


# Opening Day - Probable Starting Lineups
######################################################

# GAMEDAY Projected Starters (Starting 9 + 5 SP)
url = 'https://www.mlb.com/news/projected-lineups-rotations-for-every-2024-mlb-team'

# Current Day Rosters - Out
OD_rosters = []

page = requests.get(url)
soup = BeautifulSoup(page.text, 'html.parser')

# Could not easily extract team name/id
for item in soup.select('div.Styles__MarkdownContainer-dxqlie-0.eJdqRG > ol > li'):
    try:
        name = item.text.split('/')[0].strip().split(',')[0]
        pos = item.text.split(',')[1].strip()
        player_id = int(''.join(re.findall(r'\d+', item.find('a')['href'])))
        
        OD_rosters.append([name, player_id, pos])
    except:
        pass

OD_rosters = pd.DataFrame(OD_rosters).rename({0: 'name',
                                              1: 'key_mlbam',
                                              2: 'pos'}, axis = 1)



################## MANUAL ADDS ####################
# MYLES STRAW for KC doesn't have a comma for the split - added manually
# and one Team only has 8 players - mlb link doesn't have playerid
# TEAMIDs
#OD_rosters.to_csv("OD_rosters.csv", index = False)
#OD_rosters = pd.read_csv('OD_rosters.csv')

# Lookup Fangraphs ID
player_ids = pyb.playerid_reverse_lookup(OD_rosters.key_mlbam, key_type = 'mlbam')
OD_rosters = OD_rosters.merge(player_ids[['key_mlbam', 'key_fangraphs', 'key_bbref']], how = 'left', on = 'key_mlbam')


# Position Player or Pitcher?
OD_rosters['pos_type'] = 'position'
OD_rosters.loc[OD_rosters['pos'].isin(['RHP','LHP']), 'pos_type'] = 'pitcher'


#  Get teamID
OD_rosters = OD_rosters.merge(CY_WAR[['key_mlbam','teamID']], how = 'left', on = 'key_mlbam')
OD_rosters.groupby('teamID')['pos_type'].value_counts()


# First pass for missing players
OD_rosters = OD_rosters.merge(CY_rosters[['key_mlbam','teamID']], how = 'left', on = 'key_mlbam')
OD_rosters.loc[OD_rosters['teamID_x'].isnull(), 'teamID_x'] = OD_rosters['teamID_y']
del OD_rosters['teamID_y']
OD_rosters.rename({'teamID_x': 'teamID'}, axis = 1, inplace = True)


# Check for any missing matches
OD_rosters[OD_rosters['teamID'].isnull()]
a = OD_rosters.groupby('teamID')['pos_type'].value_counts()


# Find team with incomplete rosters (< 9 position players or < 5 starting pitchers)
missing_pos = a[(a.index.get_level_values('pos_type') == 'position') & (a.values < 9)].index.get_level_values('teamID').to_series()
missing_pitcher = a[(a.index.get_level_values('pos_type') == 'pitcher') & (a.values < 5)].index.get_level_values('teamID').to_series()

OD_rosters.loc[OD_rosters['teamID'].isin(missing_pos), ['teamID','pos']]
OD_rosters.loc[OD_rosters['teamID'].isin(missing_pitcher), ['teamID','pos']]


############ Export & Manually Fix  If Needed #####

# ['C','1B','2B','SS','3B','LF','CF','RF','DH']

#OD_rosters.to_csv("OD_rosters.csv", index = False)
#OD_rosters = pd.read_csv('OD_rosters.csv')

###################################################


# Add in WAR
OD_rosters = OD_rosters.merge(CY_WAR[['key_mlbam','WAR']], how = 'left', on = 'key_mlbam')


# Impute missing WAR
missing_WAR = OD_rosters[OD_rosters.WAR.isnull()]
missing_WAR = missing_WAR.sort_values('name')
OD_rosters = OD_rosters[~OD_rosters.WAR.isnull()]

impute_WAR = CY_WAR.loc[CY_WAR['player'].isin(missing_WAR['name'])][['player','WAR']]
impute_WAR = impute_WAR.sort_values('player')

missing_WAR['WAR'] = list(impute_WAR['WAR'])

OD_rosters = pd.concat([missing_WAR, OD_rosters])
OD_rosters['WAR'] = pd.to_numeric(OD_rosters['WAR'])

OD_rosters[OD_rosters['WAR'].isnull()]
OD_rosters.groupby('teamID')['pos_type'].value_counts()



# =============================================================================
# TEAM Win/Loss
# =============================================================================

# Previous Year
######################################################

# BATTING (Runs Scored)
PY_team_bat = pyb.team_batting(yrs[0], max(yrs))
PY_team_bat.sort_values(by = ['Team','Season'], inplace = True)
PY_team_bat = PY_team_bat.reset_index(drop = True)


# PITCHING (Runs Allowed)
PY_team_pitch = pyb.team_pitching(yrs[0], max(yrs))
PY_team_pitch.sort_values(by = ['Team','Season'], inplace = True)
PY_team_pitch = PY_team_pitch.reset_index(drop = True)


# Games Won/Lost
PY_standings = sapi.standings_data(season = max(yrs))
PY_WL = []

for item in PY_standings:
    for team in PY_standings[item]['teams']:
        PY_WL.append(team)

PY_WL = pd.DataFrame(PY_WL).rename({'team_id': 'teamID',
                                    'w': 'PY_W',
                                    'l': 'PY_L'}, axis = 1)

PY_WL['PY_W_PCT'] = PY_WL.PY_W / (PY_WL.PY_W + PY_WL.PY_L)
PY_WL = PY_WL.merge(team_master[['teamID','teamCode']], how = 'left', on = 'teamID')



# =============================================================================
# CY PYT
# =============================================================================

team_master.head()
list(PY_WL.columns)
list(CL_RS.columns)
list(CL_RA.columns)
list(PY_WAR_bat.columns)
list(PY_WAR_pitch.columns)
list(CY_WAR.columns)
list(OD_rosters.columns)


dat = PY_WL[['teamID','PY_W','PY_W_PCT']]
dat['PY_W_PCT'] = round(dat['PY_W_PCT'], 3)
dat = dat.merge(team_master[['teamID','teamIDBR','teamName']], how = 'left', on = 'teamID')
dat = dat.merge(CL_RS[['teamID','RS','CL_RS_ADJ']], how = 'left', on = 'teamID')
dat = dat.merge(CL_RA[['teamID','RA','CL_RA_ADJ']], how = 'left', on = 'teamID')
dat['PY_PYT_W_PCT'] = round((dat['RS'] ** 1.83) / ((dat['RS'] ** 1.83) + (dat['RA'] ** 1.83)), 3)
dat['PY_PYT_W'] = round(dat['PY_PYT_W_PCT'] * 162)
dat['PY_PYT_W_DIFF'] = dat['PY_PYT_W'] - dat['PY_W']

dat['PY_RS_NEW'] = dat['RS'] + dat['CL_RS_ADJ']
dat['PY_RA_NEW'] = dat['RA'] + dat['CL_RA_ADJ']
dat['PY_RS_CHG'] = dat['PY_RS_NEW'] - dat['RS']
dat['PY_RA_CHG'] = dat['PY_RA_NEW'] - dat['RA']

PY_team_WAR_bat = PY_players_bat.groupby('teamID')['WAR'].sum().reset_index().rename({'WAR': 'PY_WAR_bat'}, axis = 1)
PY_team_WAR_SP = PY_players_SP.groupby('teamID')['WAR'].sum().reset_index().rename({'WAR': 'PY_WAR_SP'}, axis = 1)
PY_team_WAR_RP = PY_players_RP.groupby('teamID')['WAR'].sum().reset_index().rename({'WAR': 'PY_WAR_RP'}, axis = 1)
PY_team_WAR_RP = PY_team_WAR_RP.merge(team_master[['teamID','lgID']], how = 'left', on = 'teamID')
PY_team_WAR_RP = PY_team_WAR_RP.merge(LG_RP_WAR, how = 'left', on = 'lgID')
PY_team_WAR_RP['RP_RA_ADJ'] = round((PY_team_WAR_RP['PY_WAR_RP'] - PY_team_WAR_RP['WAR_pitch']) * 10)

OD_rosters.groupby('teamID')['pos_type'].value_counts()

CY_team_WAR_bat = OD_rosters[OD_rosters['pos_type'] == 'position'].groupby('teamID')['WAR'].sum().reset_index().rename({'WAR': 'CY_WAR_bat'}, axis = 1)
CY_team_WAR_SP = OD_rosters[OD_rosters['pos_type'] == 'pitcher'].groupby('teamID')['WAR'].sum().reset_index().rename({'WAR': 'CY_WAR_SP'}, axis = 1)

WARs = PY_team_WAR_bat.copy()
WARs = WARs.merge(PY_team_WAR_SP, on = 'teamID')
WARs = WARs.merge(CY_team_WAR_bat, on = 'teamID')
WARs = WARs.merge(CY_team_WAR_SP, on = 'teamID')
WARs['BAT_RS_ADJ'] = round((WARs.CY_WAR_bat - WARs.PY_WAR_bat) * 10)
WARs['SP_RA_ADJ'] = round((WARs.CY_WAR_SP - WARs.PY_WAR_SP) * 10) * -1
WARs = WARs.merge(PY_team_WAR_RP[['teamID','RP_RA_ADJ']], on = 'teamID')
dat = dat.merge(WARs[['teamID','CY_WAR_bat','CY_WAR_SP','BAT_RS_ADJ','SP_RA_ADJ','RP_RA_ADJ']], on = 'teamID')

dat['CY_RS_NEW'] = dat['RS'] + dat['CL_RS_ADJ'] + dat['BAT_RS_ADJ']
dat['CY_RA_NEW'] = dat['RA'] + dat['CL_RA_ADJ'] + dat['SP_RA_ADJ'] + dat['RP_RA_ADJ']


# Normalize to total RS environment
dat['CY_RS_NEW'].sum()
dat['CY_RA_NEW'].sum()
dat['CY_RA_PCT'] = dat['CY_RA_NEW'] / dat['CY_RA_NEW'].sum()
dat['CY_RA_NEW'] = round(dat['CY_RA_PCT'] * dat['CY_RS_NEW'].sum())
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
         'PY_RS_NEW',
         'PY_RS_CHG',
         'PY_RA_NEW',
         'PY_RA_CHG',
         'CY_WAR_bat',
         'CY_WAR_SP',
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


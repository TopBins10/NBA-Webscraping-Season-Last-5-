import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests

print('Traveling to NBA webstie . . . .')
chrome_options = Options()
chrome_options.add_argument("--headless")
driver = webdriver.Chrome(options=chrome_options) 
driver.get("https://www.nba.com/players")

print('Traveled to NBA webstie . . . .')
print("*clap* DAMN!")

wait = WebDriverWait(driver, 20)
driverLoad = wait.until(EC.element_to_be_clickable((By.XPATH, "//*[@aria-label='Show Historic Toggle']")))

allPages = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "[title='Page Number Selection Drown Down List']")))
allPagesClick = Select(allPages)
allPagesSel = allPagesClick.select_by_visible_text("All")

print("/\/\ All players shown! \/\/")

driverLoad

nbaPageSource = driver.page_source

driverLoad

driver.quit()

soup = BeautifulSoup(nbaPageSource, 'lxml')

playersBox = soup.find('table', class_='players-list')
playersRow = playersBox.find('tbody').find_all('td', class_ = "primary text RosterRow_primaryCol__1lto4")

playerPgSrc = f"{playersRow}"

soup = BeautifulSoup(playerPgSrc, 'lxml')

playerLinks = soup.find_all('a', class_= 'Anchor_anchor__cSc3P RosterRow_playerLink__qw1vG')

playerLinksArr = []

for link in playerLinks:
    href = link.get('href')
    href = f"{href}"
    playerLinksArr.append(href)
    
playerFound = False

while not playerFound:
    desiredFirst = input("Player First Name: ")
    desiredLast = input("Player Last Name: ")
    playerFullName = f"{desiredFirst}-{desiredLast}".lower()
    for index in playerLinksArr:
        if playerFullName in index:
            print("Player found")
            extension = index
            print(extension)
            playerFound = True
            break
    if not playerFound:
        print("Does not exist")

playerStatWeb = requests.get(f"https://www.nba.com{extension}").text
soup = BeautifulSoup(playerStatWeb, 'lxml')
statsFind = soup.find_all('div', class_='PlayerSummary_playerStat__rmEOP')
statsHTML = BeautifulSoup(f"{statsFind}", 'lxml')
statsNames = statsHTML.find_all('p', 'PlayerSummary_playerStatLabel__I3TO3')
statsNums = statsHTML.find_all('p', 'PlayerSummary_playerStatValue___EDg_')

statNameList = [name.text for name in statsNames]
statNumsList = [num.text for num in statsNums]

seasonStats = pd.DataFrame([statNumsList], columns=statNameList)
print('Season Stats:')
print(seasonStats.to_string(index=False))
seasonStats.to_excel('seasonStats.xlsx', index = False)
print('shared to excel')

soup = BeautifulSoup(playerStatWeb, 'lxml')

stats5Find = soup.find_all('div', class_='MockStatsTable_statsTable__V_Skx')

soup = BeautifulSoup(f"{stats5Find}", 'lxml')

stats5Name = soup.find_all('th')
stat5NameList = [name.text for name in stats5Name]

last5 = soup.find('tbody').find_all('tr')
gameArr = []

for game in last5:
    gameData = []
    data = game.find_all('td')
    for dataPoints in data:
        value = dataPoints.text.strip()
        gameData.append(value)
    gameArr.append(gameData)

headers = stat5NameList
rows = gameArr

last5data = pd.DataFrame(rows,columns = headers) # creates the last 5 games data tabble

last5data.loc[last5data['W/L'] == 'W', 'W/L'] = '1' # won
last5data.loc[last5data['W/L'] == 'L', 'W/L'] = '0' # lost
last5data.loc[last5data['W/L'] == '', 'W/L'] = '2' # currently playing

# Update 'Matchup' to '0' for away games (if '@' is in the string)
last5data['Matchup'] = last5data['Matchup'].apply(lambda x: '0' if '@' in x else x)

# Update 'Matchup' to '1' for home games (if 'vs' is in the string)
last5data['Matchup'] = last5data['Matchup'].apply(lambda x: '1' if 'vs' in x else x)

last5data['Game Date'] = range(1, len(last5data) + 1)

last5data.to_excel('last5.xlsx', index = False)
print(last5data.to_string(index=False))
print('shared to excel')



    
    

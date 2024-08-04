from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd
import time

# Set the path to the Chrome driver executable
service = Service('/Users/kotaamemiya/Desktop/chromedriver-mac-arm64/chromedriver')
options = webdriver.ChromeOptions()
options.add_argument('--headless')
driver = webdriver.Chrome(service=service, options=options)

url = 'https://www.slickcharts.com/sp500'
driver.get(url)

time.sleep(5)

table = driver.find_element(By.CLASS_NAME, 'table-hover')

symbols = []
rows = table.find_elements(By.TAG_NAME, 'tr')
for row in rows[1:]:
    cells = row.find_elements(By.TAG_NAME, 'td')
    if cells:
        symbols.append(cells[2].text.strip())

driver.quit()

df = pd.DataFrame(symbols, columns=['Symbol'])
df.to_csv('sp500_symbols.csv', index=False, header=False)

print("CSV file 'sp500_symbols.csv' created successfully!")

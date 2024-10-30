import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

s = Service("D:\\Desktop\\chromedriver.exe")
driver = webdriver.Chrome(service=s)

all_html_content = []

for i in range(1, 50):
    try:
        driver.get(f'https://www.bproperty.com/buy/dhaka/?page={i}')
        
        # Scroll to the bottom to load more content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        
        # Wait for a short duration to ensure content loads
        time.sleep(2)  # Adjust this value based on the page loading speed
        
        # Get the page source
        page_source = driver.page_source
        all_html_content.append(page_source)  # Collect the page source
        
    except Exception as e:
        print("An error occurred on page", i, ":", e)

# Save the accumulated HTML content to a single file
with open('bproperty_all_pages.html', 'w', encoding='utf-8') as f:
    f.write(''.join(all_html_content))  # Join the list into a single string

driver.quit()

import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import pandas as pd

# Configure the webdriver
s = Service("D:\\Desktop\\chromedriver.exe")
driver = webdriver.Chrome(service=s)

# List to store property data
all_property_data = []

# Define the number of pages to scrape
num_pages = 50  # Change this to the desired number of pages

# Scrape the specified number of pages
for i in range(1, num_pages + 1):
    try:
        # Navigate to the page
        driver.get(f'https://www.bproperty.com/buy/dhaka/mirpur/?page={i}')
        
        # Scroll to the bottom to load more content
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)  # Wait for the page to load
        
        # Find all listing containers on the page
        listings = driver.find_elements(By.CLASS_NAME, 'js-MainListings-container')
        
        for listing in listings:
            try:
                # Extract data from each listing
                property_data = {
                    "property_name": listing.find_element(By.CLASS_NAME, "ListingCell-KeyInfo-title").text if listing.find_elements(By.CLASS_NAME, "ListingCell-KeyInfo-title") else None,
                    "address": listing.find_element(By.CLASS_NAME, "ListingCell-KeyInfo-address-text").text if listing.find_elements(By.CLASS_NAME, "ListingCell-KeyInfo-address-text") else None,
                    "short_description": listing.find_element(By.CLASS_NAME, "ListingCell-shortDescription").text if listing.find_elements(By.CLASS_NAME, "ListingCell-shortDescription") else None,
                    "price": listing.find_element(By.CLASS_NAME, "PriceSection-FirstPrice").text if listing.find_elements(By.CLASS_NAME, "PriceSection-FirstPrice") else None,
                    "bedrooms": listing.find_element(By.CLASS_NAME, "icon-bedrooms").text if listing.find_elements(By.CLASS_NAME, "icon-bedrooms") else None,
                    "bathrooms": listing.find_element(By.CLASS_NAME, "icon-bathrooms").text if listing.find_elements(By.CLASS_NAME, "icon-bathrooms") else None,
                    "floor_area": listing.find_element(By.CLASS_NAME, "icon-livingsize").text if listing.find_elements(By.CLASS_NAME, "icon-livingsize") else None,
                    "property_url": listing.find_element(By.CLASS_NAME, "js-listing-link").get_attribute("href"),
                    "property_type": "buy"
                }
                
                # Add to the list of properties
                all_property_data.append(property_data)
            except Exception as e:
                print(f"Error scraping a listing on page {i}: {e}")
    
    except Exception as e:
        print(f"An error occurred on page {i}: {e}")

# Save the data to a CSV file
df = pd.DataFrame(all_property_data)
df.to_csv("mirpur_properties.csv", index=False)
print("Data saved to all_properties.csv")

# Close the driver
driver.quit()

from selenium import webdriver  # pip install selenium
import time

from selenium.webdriver.chrome.webdriver import WebDriver




chrome = "C:/New folder/chromedriver.exe" ###https://chromedriver.chromium.org/

email = "pythonoynani@gmail.com"
password = "vivek chary"


def login():
    driver: WebDriver = webdriver.Chrome(chrome)
    driver.get("https://accounts.google.com/ServiceLogin/identifier?service=mail&passive=1209600&osid=1&continue=https%3A%2F%2Fmail.google.com%2Fmail%2Fu%2F0%2F&followup=https%3A%2F%2Fmail.google.com%2Fmail%2Fu%2F0%2F&emr=1&flowName=GlifWebSignIn&flowEntry=ServiceLogin")
    driver.find_element_by_xpath("/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[1]/div/form/span/section/div/div/div[1]/div/div[1]/div/div[1]/input").send_keys(email)
    driver.find_element_by_xpath("/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[2]/div/div[1]/div/div/button/span").click()
    time.sleep(5)
    driver.find_element_by_xpath("/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[1]/div/form/span/section/div/div/div[1]/div[1]/div/div/div/div/div[1]/div/div[1]/input").send_keys(password)
    driver.find_element_by_xpath("/html/body/div[1]/div[1]/div[2]/div/div[2]/div/div/div[2]/div/div[2]/div/div[1]/div/div/button/span").click()
    time.sleep(30)
    driver.get("https://mail.google.com/mail/u/0/#inbox/FMfcgzGmvLMGkhjWbPVqHNsgzNDCVbrN")
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div[2]/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/table/tbody/tr/td[2]/div/div/span[1]").click()
    time.sleep(10)
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div[2]/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/div[4]/table/tbody/tr/td[2]/table/tbody/tr[1]/td/div/div[2]/div[1]/div/table/tbody/tr/td[2]/div[2]/div/div[1]").send_keys("hello, how are you ")
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/div[4]/table/tbody/tr/td[2]/table/tbody/tr[2]/td/div[2]/div[1]/div[4]/table/tbody/tr/td[1]/div/div[2]/div[1]").click()
    time.sleep(60)
    driver.find_element_by_xpath("/html/body/div[4]/span[2]").click()
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div[2]/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/table/tbody/tr/td[2]/div/div/span[1]").click()
    time.sleep(10)
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div[2]/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/div[4]/table/tbody/tr/td[2]/table/tbody/tr[1]/td/div/div[2]/div[1]/div/table/tbody/tr/td[2]/div[2]/div/div[1]").send_keys("thank you for asking i am good btw wt help do you want from us")
    driver.find_element_by_xpath("/html/body/div[7]/div[3]/div/div[2]/div[1]/div[2]/div/div/div/div/div[2]/div/div[1]/div/div/div/table/tr/td[1]/div[2]/div[2]/div/div[3]/div/div/div/div/div/div[2]/div/div/div/div[4]/table/tbody/tr/td[2]/table/tbody/tr[2]/td/div[2]/div[1]/div[4]/table/tbody/tr/td[1]/div/div[2]/div[1]").click()
    time.sleep(30)


login()
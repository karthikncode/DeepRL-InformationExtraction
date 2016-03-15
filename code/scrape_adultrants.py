from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import pickle


int2col = ['FoodCatergory', 'YearBegan', 'YearEnded', "AdulteratedFood(s)", 'Adulterant(s)', "Method of Adulteration", "Location(s)", "Open"]
int2allFeilds = \
['Incident_Summary', \
'Potential_Public_Health_Impact', \
'Year_Began', \
'Year_Ended', \
'Number_of_Illnesses', \
'Number_of_Deaths', \
'Food_Category', \
'Primary_Method_of_Adulteration', \
'Secondary_Method_of_Adulteration', \
'Number_of_References', \
'Consumer_Brand', \
'Perpetrator', \
'Adulterated_Food_Product', \
'Affected_Food_Product', \
'Adulterant(s)', \
'Produced_Location', \
'Distributed_Location']
int2citationFeilds = ['Authors', 'Date', 'Title', 'Source']

incidents = {}

def login(driver):
	login = driver.find_element_by_name("login")
	password = driver.find_element_by_name("password")

	login.send_keys('yygu@mit.edu')
	password.send_keys('csail1234')
	password.send_keys(Keys.RETURN)

def getToDatabase(driver):
	login(driver)

	while "autoexit.cfm" in driver.current_url:
		relogin = driver.find_element_by_partial_link_text("Click Here")
		relogin.click()
		login(driver)

	driver.get("https://www.foodshield.org//member/apps/redirect.cfm?appid=0D31273B-F1F6-41D7-9BC8-FB10F1A6716B")

	database = driver.find_element_by_partial_link_text("FPDI EMA Incidents Database")
	database.click()
	driver.find_element_by_name('commit').click()

def get_id(elem):
	btn = elem.find_element_by_css_selector('a.btn')
	incident_id = btn.get_attribute("href").split("/")[-1]
	return incident_id, btn

def scrapeIncidentPage(driver, btn, incident_id):
	btn.send_keys(Keys.RETURN)
	incidents[incident_id] = {}
	section = driver.find_element_by_css_selector("section")
	feilds = section.find_elements_by_css_selector("p")
	for f_ind, f in enumerate(feilds):
		incidents[incident_id][int2allFeilds[f_ind]] = f.text
	citations = []
	table = section.find_element_by_css_selector("table.table")
	tbody = table.find_element_by_css_selector("tbody")
	rows  = tbody.find_elements_by_css_selector("tr")
	for row in rows:
		cols = row.find_elements_by_css_selector("td")
		citation = {}
		for col_ind, col in enumerate(cols[:-1]):
			citation[int2citationFeilds[col_ind]] = col.text
		citations.append(citation)
	incidents[incident_id]["citations"] = citations
	driver.back()

def setupUpSearchPage(driver):
	table = driver.find_element_by_id('DataTables_Table_0')
	body = table.find_element_by_css_selector("tbody")
	rows = body.find_elements_by_css_selector("tr")
	return rows

def scrapeSearchPage(driver, page_ind):
	rows =setupUpSearchPage(driver)
	for row_ind in range(len(rows)):
		row = rows[row_ind]
		cols = row.find_elements_by_css_selector("td")
		incident_id, btn = get_id(cols[-1])
		scrapeIncidentPage(driver, btn, incident_id)
		print 'scraped incident:', (page_ind*100 + row_ind)
		rows = setupUpSearchPage(driver)

driver = webdriver.Firefox()
driver.get('https://www.foodshield.org/member/login')
getToDatabase(driver)
scrapeSearchPage(driver, 0)
for i in range(5):
	nextPage = driver.find_element_by_partial_link_text("Next")
	nextPage.click()
	scrapeSearchPage(driver, i+1)

pickle.dump(incidents, open('EMA_dump.p', 'wb'))

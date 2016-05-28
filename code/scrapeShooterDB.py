from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from scrape import download_article
import time
import pickle

# Dumps incidents in a pickle file
def save(incidents):
    print "Saving incidents with ", len(incidents), "entries"
    pickle.dump(incidents, open("shooter_db_scrape_dump.cleaned.p", "wb"))


#Iterate through pages of a single DB and record all docs
def scrapeDB(driver, incidents):
    scrapeRows(driver, incidents)
    while(True):
        try:
            nextPageButton = driver.find_element_by_partial_link_text("next")
            nextPageButton.click()
            scrapeRows(driver, incidents)
        except Exception, e: #Case where we are at last page
            break
    save(incidents)
    print "scrapeDB"

#For a single page of DB, record datadri
def scrapeRows(driver, incidents):
    table = driver.find_elements_by_tag_name("table")[1]
    rows = table.find_elements_by_tag_name("tr")
    for row in rows[1:]: #Skip title row
        try:
            source = row.find_element_by_partial_link_text("Source")
        except Exception, e:
            #This is the case where there are mutiple sources. We can recover this later.
            #TODO: If data is still small recover this
            continue
        href = source.get_attribute("href")

        cols = row.find_elements_by_tag_name("td")

        city = cols[2].text
        numKilled = cols[4].text
        numWounded = cols[5].text
        incidentLink = row.find_element_by_partial_link_text("Incident")
        incidentHref = incidentLink.get_attribute("href")

        incident = {
            'city':city,
            'numKilled':numKilled,
            'numWounded':numWounded,
            'incidentHref': incidentHref
        }
        incidents[href] = incident
        
#Scrape single incident page
def scrapeShooterNames(driver, incidents):
    numIncidents = len(incidents)
    index = 1
    for source in incidents:
        print index,"/",numIncidents
        incident = incidents[source]
        incidentHref = incident["incidentHref"]
        shooterName = ""

        driver.get(incidentHref)

        try:
            content = driver.find_element_by_id("block-system-main")
        except Exception, e:
            time.sleep(1)
            try:
                content = driver.find_element_by_id("block-system-main")
            except Exception, e:
                incidents[source]["shooterName"] = ""
                continue

        participants = content.find_elements_by_tag_name("div")[1]

        particpantsDetails = participants.find_elements_by_tag_name("ul")

        for memeber in particpantsDetails:
            details = memeber.text
            if  "Victim" in details:
                continue

            detailList = details.split("\n")

            nameText = [d for d in detailList if "Name:" in d]
            if len(nameText) >0:
                nameText = nameText[0]
            else:
                continue
            shooterName += "|".join(nameText.split()[1:])
        incidents[source]["shooterName"] = shooterName
        index += 1


def downloadArticles(incidents):
    numIncidents = len(incidents)
    delSet = []
    index = 1
    for source in incidents:
        if "title" in incidents[source]:
            index += 1
            continue
        else:
            # print incidents[source]
            print index,"/",numIncidents
            results = download_article(source, False, True)
            if results[0]:
                success, title, text, date, title2 = results
                incidents[source]["title"] = title
                incidents[source]["body"]  = text
                incidents[source]["publishDate"]  = date
            else:
                delSet.append(source)
            index += 1    
    print len(incidents)

    print "Number of keys that are have no articles", len(delSet)
    for source in delSet:
        del incidents[source]
    print len(incidents)        

## Script
## Go through DBs at each link, and collect :Title, Killer, City, Num killed,
# num wounded, and source article link for where this happened in some python dict

if __name__ == "__main__":
    incidents = {}
    baseurl="http://www.gunviolencearchive.org/"
    urls = ['children-killed', 'children-injured', 'teens-killed', \
        'teens-injured', 'accidental-deaths', 'accidental-injuries', \
        'officer-involved-shootings']



    incidents = pickle.load(open("shooter_db_scrape_dump.cleaned.p", "rb"))
    # Iterate through different DB Sites
    if False: #not rescraping incidents
        driver = webdriver.Firefox()
        for path in urls:
            url = baseurl + path
            driver.get(url)
            scrapeDB(driver, incidents)
        scrapeShooterNames(driver, incidents)

    downloadArticles(incidents)
    save(incidents)



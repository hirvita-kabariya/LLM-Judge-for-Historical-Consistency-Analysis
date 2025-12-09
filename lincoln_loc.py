import os
import json
import pathlib
import time
import requests

#A list of dictionaries for storing LoC metadata
loc_items = [
    {
        "id": "mal0440500", 
        "title": "Election Night 1860 Letter",
        "url": "https://tile.loc.gov/storage-services/service/mss/mal/044/0440500/0440500.xml",
        "document_type": "Letter",
        "date": "November 10, 1860",
        "place": "Springfield, IL",
        "from": "Abraham Lincoln",
        "to": "Truman Smith", 
    },
    {
        "id": "mal.0882800",
        "title": "Fort Sumter Decision",
        "url": "https://tile.loc.gov/storage-services/service/mss/mal/088/0882800/0882800.xml",
        "document_type": "Letter",
        "date": "April 08, 1861",
        "place": "Charleston, SC",
        "from": "Robert S. Chew",
        "to": "Abraham Lincoln",
    },
    {
        "id": "gettysburg_nicolay_copy",
        "title": "Gettysburg Address",
        "url": "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
        "document_type": "Speech",
        "date": "November 19, 1863", 
        "place": "Gettysburg, PA",
        "from": "Abraham Lincoln",
        "to": "Public Address",
    },
    {
        "id": "mal.4361300",
        "title": "Second Inaugural Address",
        "url": "https://tile.loc.gov/storage-services/service/mss/mal/436/4361300/4361300.xml",
        "document_type": "Speech",
        "date": "March 4, 1865",
        "place": "Washington, D.C.",
        "from": "Abraham Lincoln",
        "to": "Public Address",
    },
    {
        "id": "mal.4361800",
        "title": "Last Public Address",
        "url": "https://tile.loc.gov/storage-services/service/gdc/gdccrowd/mss/mal/436/4361800/4361800.txt",
        "document_type": "Letter",
        "date": "March 18, 1865",
        "place": "Wheeling, WV",
        "from": "E. B. Hall",
        "to": "Abraham Lincoln and James Speed",
    },
]


def ensure_dir(path: str) -> None:
    #create a folder for storing json files and downloaded books
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def download_loc_pages(items, raw_dir: str) -> None:

    ensure_dir(raw_dir)
    for i in items:
        url = i["url"]
        loc_id = i["id"]
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        #save pages as .html
        out_path = os.path.join(raw_dir, f"{loc_id}.html")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        #to respect LoC API rate limits
        time.sleep(1)


#converting downloaded .html page to JSON items
def build_loc_dataset(items, raw_dir: str):

    dataset = []

    for i in items:
        loc_id = i["id"]
        raw_path = os.path.join(raw_dir, f"{loc_id}.html")

        with open(raw_path, "r", encoding="utf-8") as f:
            content = f.read()

        item = {
            "id": f"loc_{loc_id}",          
            "title": i["title"],
            "reference": i["url"],       
            "document_type": i["document_type"],
            "date": i["date"],           
            "place": i["place"],
            "from": i["from"],
            "to": i["to"],
            "content": content,           
        }

        dataset.append(item)

    return dataset

#save list of JSON items into one JSON dataset file
def save_loc_dataset(items, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

#Location to download loc pages and json dataset
raw_dir = os.path.join("data", "raw", "loc")
out_json = os.path.join("data", "final_dataset", "lincoln_loc.json")


download_loc_pages(loc_items, raw_dir)
items = build_loc_dataset(loc_items, raw_dir)
save_loc_dataset(items, out_json)


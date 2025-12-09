import os
import json
import pathlib
import requests

#A list of dictionaries for storing books metadata
gutenberg_books_info = [
    {
        "id": "6812",
        "title": "Abraham Lincoln: a History — Volume 01",
        "author": "John G. Nicolay and John Hay",
        "date": "Nov 1, 2004",
        "reference_url": "https://www.gutenberg.org/ebooks/6812",
    },
    {
        "id": "6811",
        "title": "The Life of Abraham Lincoln",
        "author": "Henry Ketcham",
        "date": "Nov 1, 2004",
        "reference_url": "https://www.gutenberg.org/ebooks/6811",
    },
    {
        "id": "12801",
        "title": "Abraham Lincoln, Volume II",
        "author": "John T. Morse, Jr.",
        "date": "Jul 1, 2004",
        "reference_url": "https://www.gutenberg.org/ebooks/12801",
    },
    {
        "id": "14004",
        "title": "The Every-day Life of Abraham Lincoln",
        "author": "Francis F. Browne",
        "date": "Nov 10, 2004",
        "reference_url": "https://www.gutenberg.org/ebooks/14004",
    },
    {
        "id": "18379",
        "title": "Abraham Lincoln",
        "author": "Lord Charnwood",
        "date": "May 11, 2006",
        "reference_url": "https://www.gutenberg.org/ebooks/18379",
    },
]

#URL for download
text_url = "https://www.gutenberg.org/ebooks/{id}.txt.utf-8"


def ensure_dir(path: str) -> None:
    #A folder for storing JSON files and downloaded books
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def download_books(book_list, raw_dir: str) -> None:

    ensure_dir(raw_dir)

    for book in book_list:
        book_id = book["id"]
        url = text_url.format(id=book_id)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        out_path = os.path.join(raw_dir, f"{book_id}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(response.text)

#converting the downloaded .txt book file into JSON items
def build_book_dataset(book_list, raw_dir: str):

    items = []

    for book in book_list:
        book_id = book["id"]
        txt_path = os.path.join(raw_dir, f"{book_id}.txt")

        with open(txt_path, "r", encoding="utf-8") as f:
            content = f.read()

        item = {
            "id": f"book_{book_id}",          
            "title": book["title"],               
            "reference": book["reference_url"],   
            "document_type": "Book",              
            "date": book["date"],                         
            "place": None,                        
            "from": book["author"],               
            "to": None,                           
            "content": content,                   
        }
        items.append(item)

    return items

#Saving the list of JSON items into a single JSON dataset file
def save_book_dataset(items, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


#The location used for downloading book files
raw_dir = os.path.join("data", "raw", "books")
out_json = os.path.join("data", "final_dataset", "gutenberg_authors.json")


download_books(gutenberg_books_info, raw_dir)
items = build_book_dataset(gutenberg_books_info, raw_dir)
save_book_dataset(items, out_json)


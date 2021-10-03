from bs4 import BeautifulSoup
import requests
import os
import time


def find_anime_names(startPage, endPage):
    anime_names = []

    for i in range(startPage, endPage+1):
        url = "https://www.anime-planet.com/anime/all"
        if i > 1:
            url = url + "?page=" + str(i)

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
        response = requests.get(url=url, headers=headers)
        soup = BeautifulSoup(response.text, "lxml")

        results = soup.find_all("a")
        for result in results:
            string = result.get("class")
            if string != None and len(string[0]) > 6 and string[0] == "tooltip" and len(string) > 1 and len(string[1]) > 4 and string[1][0:5] == "anime":
                name = result.get("href").split('/')[2]
                anime_names.append(name)

    return anime_names


def get_imgs(anime_names):
    for anime_name in anime_names:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.62 Safari/537.36'}
        response = requests.get(
            url=f"https://www.anime-planet.com/anime/" + anime_name + "/characters", headers=headers)
        soup = BeautifulSoup(response.text, "lxml")

        results = soup.find_all("img")

        img_links = [r.get("src") for r in results]

        for index, link in enumerate(img_links):
            if "https://" not in link:
                img = requests.get(
                    url="https://www.anime-planet.com" + link)
                with open("anime_data/" + anime_name + str(index) + ".jpg", "wb") as file:
                    file.write(img.content)
                    print(anime_name + str(index) + ".jpg")


def dataset(startPage, endPage):
    anime_names = find_anime_names(startPage, endPage)
    get_imgs(anime_names)

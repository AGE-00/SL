import os
import requests
from bs4 import BeautifulSoup

page_url = "https://duckduckgo.com/?q=%E4%BF%A1%E5%8F%B7%E6%A9%9F+%E6%AD%A9%E8%A1%8C%E8%80%85%E3%80%80%E6%97%A5%E6%9C%AC&t=brave&iar=images&iax=images&ia=images"
res = requests.get(page_url)
soup = BeautifulSoup(res.text)
print(soup.find_all("img"))

img_tags = soup.find_all("img")
img_urls = []

for img_tag in img_tags:
    img_url = img_tag.get("src")
    if img_url != None:
        img_urls.append(img_url)
        
download_folder = "download"

for i, img_url in enumerate(img_urls):
    img = requests.get(img_url, stream=True)
    
    # ファイル名を番号順にするために、インデックスを利用
    img_name = f"image_{i}.jpg"
    
    # ダウンロードフォルダに保存するパスを作成
    save_path = os.path.join(download_folder, img_name)
    
    with open(save_path, "wb") as f:
    	f.write(img.content)
    	print(f"画像をダウンロードしました: {save_path}")
import requests
import json

if __name__ == "__main__":
    esp_url = "http://192.168.0.50/?r215g255b206&"

    r = requests.get(url=esp_url)

    print(r)

def sendRGBvalues(base_ip, colors):
    # Construct JSON request
    json_colors = json.dumps(colors)
    print(json_colors)
    #return requests.get(url=base_ip+'/?r{}g{}b{}&'.format(r, g, b))
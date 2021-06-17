import requests

api_url = 'http://www.codestem.co.uk/hub/api'

r = requests.post(api_url + '/users',
    headers={
             'Authorization': 'token %s' % "916b4372b8b9499fb0f043f9ad5f9e21",
            }
    )

r.raise_for_status()
users = r.json()
print(users)



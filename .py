import requests
import urllib3
import pandas as pd
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

auth_url = "https://www.strava.com/oauth/token"

# You need to update this information from your Strava account before you can run the rest of this code
# See tutorial video here for info on how: https://www.youtube.com/watch?v=sgscChKfGyg&list=PLO6KswO64zVvcRyk0G0MAzh5oKMLb6rTW

payload = {
    'client_id': "XXXXX",
    'client_secret': '11111yyyyyy',
    'refresh_token': '22222zzzzzz',
    'grant_type': "refresh_token",
    'f': 'json'
}

print("Requesting Token...\n")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']
print("Access Token = {}\n".format(access_token))

# Now we need to import a list of activities and some top level stats about them.
# This will be 1 row per activity.

# Initialize the dataframe
col_names = ['id','type', 'name', 'distance', 'moving_time', 'elapsed_time', 'total_elevation_gain', 'start_date',  'start_latlng', 'kilojoules', 'average_heartrate', 'max_heartrate', 'elev_high', 'elev_low', 'average_speed', 'max_speed']
activities = pd.DataFrame(columns=col_names)

activites_url = "https://www.strava.com/api/v3/athlete/activities"
header = {'Authorization': 'Bearer ' + access_token}

page = 1
per_page = 50

while True:
    
    # get page of activities from Strava
    param = {'per_page': per_page, 'page': page}
    r = requests.get(activites_url, headers=header, params=param).json()

    # if no results then exit loop
    if (not r):
        break
    
    # otherwise add new data to dataframe
    for x in range(len(r)):
      for c in col_names:
        try:
          activities.loc[x + (page-1)*50, c] = r[x][c]
        except:
          activities.loc[x + (page-1)*50, c] = 'null'

    # increment page
    page += 1

print("Activites imported")
print(activities)

activities.to_csv('activities.csv')
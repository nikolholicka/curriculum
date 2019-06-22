
# Folium - Codealong

## Introduction 

In this codealong, we'll take a look at how to create an interactive map using the Folium package. From there, we'll return to APIs in the final lab for the day where you'll make an interactive map from your requests to the API!

## Objectives

You will be able to: 
* Create maps with Folium

## Creating a Basemap

Here we'll take a look at creating a basemap over the London region!


```python
import folium

lat = 51.51
long = -0.14

#Create a map of the area
base_map = folium.Map([lat, long], zoom_start=13)
base_map
```

## Adding Markers to the Map

Great! Now let's take a look at adding little markers to our map!

**Note:** you may have to zoom out to see all of the markers!


```python
import numpy as np

#Generate some random locations to add to our map
x = [lat + np.random.uniform(-.1,.1) for i in range(20)]
y = [long + np.random.uniform(-.1,.1) for i in range(20)]
points = list(zip(x, y))
for p in points:
    lat = p[0]
    long = p[1]
    marker = folium.Marker(location=[lat, long])
    marker.add_to(base_map)
base_map
```

## Adding Pop-up Boxes to Our Markers

Often we may wish to not only place markers on the map, but to create interactive pop-ups which display information to that location. To do this, we can add a popup to our markers when adding them to the map! 


```python
for p in points:
    lat = p[0]
    long = p[1]
    popup_text = "Latitude: {}, Longitude: {}".format(lat,long)
    popup = folium.Popup(popup_text, parse_html=True)
    marker = folium.Marker(location=[lat, long], popup=popup)
    marker.add_to(base_map)
base_map
```

Now, if you click on the map markers, you should see a little information box pop up!

## Summary 

In this codealong, we learned how to use Folium to create some cool interactive maps with only a few lines of python code! In the next lab, you'll synthesize your skills for the day together and create an interactive visualization map for data you retrieve from the Yelp API!

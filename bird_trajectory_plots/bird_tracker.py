# bird tracker
"""
Created on Wed Aug 15 13:07:01 2018

@author: fnarbona
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature

bird_data = pd.read_csv("bird_tracking.csv")
bird_data.info
bird_names = pd.unique(bird_data.bird_name)

## flight trajectory by bird name
#plt.figure()
#for name in bird_names:
#    ix = bird_data.bird_name == name
#    x,y = bird_data.longitude[ix], bird_data.latitude[ix]
#    plt.plot(x,y,".", label=name)
#    plt.legend()
#    
## flight speed by bird name
#for name in bird_names:
#    ix = bird_data.bird_name == name
#    x = bird_data.speed_2d[ix]
#    y = np.isnan(x)
#    plt.figure(figsize=(5,1))
#    plt.hist(x[~y], bins=np.linspace(0,30,20), normed=True, label=name)
#    plt.legend()
#
## using pandas to plot flight speed
#plt.figure()
#bird_data.speed_2d.plot(kind='hist', range=[0,30])
#plt.xlabel("2d speed")
#
#
## parsing datetime values
#timestamp = []
#for k in range(len(bird_data)):
#    timex = bird_data.date_time.iloc[k][:-3]
#    timestamp.append(datetime.datetime.strptime(timex, "%Y-%m-%d %H:%M:%S"))
#
#bird_data["timestamp"] = pd.Series(timestamp, index = bird_data.index)
#times = bird_data.timestamp[bird_data.bird_name == "Eric"]
#elapsed_time = [time - times[0] for time in times]
#elapsed_days = np.array(elapsed_time)/datetime.timedelta(days=1)
#
## elapsed days
#plt.figure()
#plt.plot(elapsed_days, label="Eric")
#plt.xlabel("observation")
#plt.ylabel("elapsed time (days)")
#plt.legend()
#
## daily mean speed
#next_day = 1
#inds = []
#mean_speed = []
#for i,t in enumerate(elapsed_days):
#        if t<next_day:
#            inds.append(i)
#        else:
#            mean_speed.append(np.mean(bird_data.speed_2d[inds]))
#            next_day +=1
#            inds = []
#    
#plt.figure()
#plt.plot(mean_speed)
#plt.xlabel("day")
#plt.ylabel("mean speed (m/s)")

proj = ccrs.Mercator()

plt.figure(figsize=(10,10))
ax = plt.axes(projection=proj)
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.set_extent((-25,20,52,10))

for name in bird_names:
    ix = bird_data.bird_name == name
    x,y = bird_data.longitude[ix], bird_data.latitude[ix]
    ax.plot(x,y,'.', transform=ccrs.Geodetic(), label=name)
    
plt.legend(loc=2)








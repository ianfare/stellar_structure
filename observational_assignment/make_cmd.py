import csv,math
import matplotlib.pyplot as plt

# Distance to target, in pc (for calculating absolute magnitudes):
dist = 1900

# Grab the obj270 data, and trim the waste spaces
# obj720 is i
obj720_data = []
with open("./obj720.phot") as csvfile:
  csv_reader = csv.reader(csvfile,delimiter = " ")
  for row in csv_reader:
    obj720_data.append(row)

for i,row in enumerate(obj720_data):
  for j in range(3):
    row.remove("")
  for j,col in enumerate(row):
    try:
      row[j] = float(col)
    except:
      bad_practices = True

# Grab the obj274 data, and trim the waste spaces
# obj724 is v
obj724_data = []
with open("./obj724.phot") as csvfile:
  csv_reader = csv.reader(csvfile,delimiter = " ")
  for row in csv_reader:
    obj724_data.append(row)

for i,row in enumerate(obj724_data):
  for j in range(3):
    row.remove("")
  for j,col in enumerate(row):
    try:
      row[j] = float(col)
    except:
      bad_practices = True

i_data = []
v_data = []
# Make magnitudes lists with "INDEF" in either filter
for i,star in enumerate(obj720_data):
  if not (obj720_data[i][3] == "INDEF" or obj724_data[i][3] == "INDEF"):
    i_data.append(obj720_data[i][3])
    v_data.append(obj724_data[i][3])

# Transform apparent magnitudes to absolute
#for i,app_mag in enumerate(i_data):
#  i_data[i] = app_mag - 5*(math.log10(dist) - 1)

#for i,app_mag in enumerate(v_data):
#  v_data[i] = app_mag - 5*(math.log10(dist) - 1)

# Get colours
colours = []
for i,star in enumerate(i_data):
  colours.append(v_data[i] - i_data[i])

plt.plot(colours,v_data,"ro",markersize = 1.5)
plt.axis([-4,2,17,7])
plt.xlabel("(V - I)")
plt.ylabel("V")
plt.title("(V - I) CMD of M11, Absolute Magnitudes")
plt.show()

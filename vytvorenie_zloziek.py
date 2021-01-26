import os

items = [
'30m-item58',
'30m-item59',
'30m-item60',
'30m-item62',
'30m-item63',
'30m-item64',
'30m-item65',
'30m-item67',
'30m-item68',
'30m-item69',
'30m-item70',
'30m-item71',
'30m-item72',
'30m-item73',
'30m-item74',
'30m-item75',
'30m-item76',
'30m-item77'
]

UNIT_COUNT = [
16, 
32, 
64, 128, 256, 512
]
HISTORY = [16, 
32, 64, 128, 256, 512
]
BATCH_SIZES = [
16, 
32, 64, 128, 256, 512
]

path = "/home/kriza/programing/clanok/"

for item in items:
  for i in BATCH_SIZES:
    for j in HISTORY:
      for k in UNIT_COUNT:
        if i == 512 and j == 512 and k == 512:
          continue
        else:
            univariate_past_history = j
            BATCH_SIZE = i
            UNITS = k
            try:
                os.makedirs(path + "saved_models/" + item + "/hist=" + str(univariate_past_history) + "batch=" + str(BATCH_SIZE) + "units=" + str(UNITS))
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
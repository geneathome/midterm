import numpy as np

import serial

import time


waitTime = 0.1


# generate the waveform table



song =np.array(
[
  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,

  392, 392, 349, 349, 330, 330, 294,

  392, 392, 349, 349, 330, 330, 294,

  261, 261, 392, 392, 440, 440, 392,

  349, 349, 330, 330, 294, 294, 261,
  261, 261, 294, 261, 349, 330,

  261, 261, 294, 261, 392, 349,

  261, 261, 523, 440, 349, 330,

  349, 349, 330, 261, 294, 261,
  261, 261, 523, 440, 349, 330,

  349, 349, 330, 261, 294, 261
]

)

song2 =np.array(
[
  261, 293, 329, 261, 261, 293, 329,

  261, 329, 349, 392, 369, 329, 349,

  392, 392, 392, 440, 392, 349, 329,

  261, 392, 440

]

)

song = song /499
song2 = song2 /499

# output formatter
print(song)
print(song2)

a = 1

formatter = lambda x: "%.3f" % x


# send the waveform table to K66F

serdev = '/dev/ttyACM0'

s = serial.Serial(serdev)

s.write(bytes(formatter(song[0]), 'UTF-8'))
s.write(bytes(formatter(song[0]), 'UTF-8'))
s.write(bytes(formatter(song[0]), 'UTF-8'))
s.write(bytes(formatter(song[0]), 'UTF-8'))
s.write(bytes(formatter(song[0]), 'UTF-8'))


while(1):
    b = s.readline()
    print(b)
    print(b[0])
    if((b[0] == 48) or (b[0] == 49)):
      print("Sending signal ...")

      print("It may take about  seconds ..." )

      for data in song:
          s.write(bytes(formatter(data), 'UTF-8'))
          time.sleep(waitTime)

          #s.close()

      print("Signal sended")
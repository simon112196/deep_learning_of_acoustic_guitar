import os
import glob
import sys


FILE_TO_ID = {
  'a':0, 'b': 1, 'c': 2, 'd': 3, 'e':4, 'f':5, 'g':6, 'h':7, 
  'i':9, 'j':9, 'k':10, 'l':11, 'm':12, 'n':13, 'o':14, 'p':15, 'q':16, 'r':17,
  's':18, 't':19, 'u':20, 'v':21
}

FFMPEG_COMMAND = os.path.join('.', 'ffmpeg', 'bin', 'ffmpeg') + ' -i %s -ac 1 -acodec pcm_f32le -ar 44100 %s -v 1' 


def ConvertFile(audio, itr, d_out): # Name conversion  and format conversion
  parent = audio.split(os.path.sep)[-2]
  output_name = ''
  output_name = '%d_%d.wav'%(itr, FILE_TO_ID[parent])
  output_name = os.path.join(d_out, output_name)
  
  os.system(FFMPEG_COMMAND%(audio, output_name))
  return True

def Convert_to_wav(d_in, d_out=os.path.join('.', 'data', 'train')): #Collecting files from the folder provided
  if not os.path.exists(d_in):
    raise(ValueError(d_in + ' not found'))
  if not os.path.exists(d_out):
    raise(ValueError(d_out + ' not found'))

  iter_num = 0 #number of files received
  d_list = os.listdir(d_in)
  print("Convertion start")
  for d in d_list:
    d = os.path.join(d_in, d)
    audios = glob.glob(os.path.join(d, '*.*'))
    for audio in audios:
      if ConvertFile(audio, iter_num, d_out):
        iter_num += 1
        if iter_num%500==0:
          print("%d records collected"%iter_num)

     
  print("%d records collected in total"%iter_num)

  return iter_num


if __name__ == '__main__':
    Convert_to_wav(sys.argv[1])

from pydub import AudioSegment
import os
import sys
#removing silences of audio file
def edit(dir_in, dir_out=os.path.join('.', 'edited')):
    m = 0
    s = 1
    dirs = os.listdir(dir_in)
    first_cut_point = (m*60 + s)* 1000
    for d in dirs:
        parent = os.path.join(dir_in, d)
        if os.path.isfile(os.path.join(dir_in, d)):
            continue
        files = os.listdir(parent) 
        path = os.path.join(dir_out, d)
        if not os.path.isdir(path): 
            os.mkdir(path)
        for file in files:
            sound = AudioSegment.from_wav(os.path.join(parent, file))
            sound_clip = sound[first_cut_point:]
            sound_clip.export(os.path.join(path, file), format='wav')
    
    print('Editing complete')

if __name__ == '__main__':
    edit(sys.argv[1])
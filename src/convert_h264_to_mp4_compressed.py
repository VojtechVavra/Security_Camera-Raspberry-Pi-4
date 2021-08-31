from subprocess import call
import fcntl, os


# Convert the h264 format to the mp4 format.
#video_folder = "/home/pi/Videos/"
lock_file = "/home/pi/Projects/Python/lockfile.lck"
video_path = "/home/pi/Videos/"

# https://stackoverflow.com/questions/6931342/system-wide-mutex-in-python-on-linux
class Locker:
    def __enter__ (self):
        self.fp = open(lock_file, 'a')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

def load_video_names():	
	with Locker():
		with open(lock_file, 'r+') as f: 	# 'a' append, 'w' write
			data = f.read()
			return data

def delete_first_name_from_file():
	with Locker():
		data2 = ""
		with open(lock_file, 'r') as fin:
			data2 = fin.read().splitlines(True)
		with open(lock_file, 'w') as fout:
			fout.writelines(data2[1:])
			#with open(lock_file, 'w+') as f: 	# 'a' append, 'w' write
			#data2 = f.read()
			#f.writelines(data2[1:])
			#data2 = data2.splitlines()
			#for line in data2[1:]:		
			#	f.write(line + '\n')

def convert_video_mp4(video_name, output_filename):
	command1 = "ffmpeg -i " + video_path + video_name + " -vcodec libx264 -crf 28 " + video_path + output_filename
	call([command1], shell=True)
def delete_video(video_name):
	rm_command1 = "rm " + video_path + video_name
	call([rm_command1], shell=True)
	
data = ""	
data = load_video_names() # loads all video names for convert
data = data.splitlines()

for video_name in data:		# loop through all video names
	output_filename = os.path.splitext(video_name)[0]
	output_filename += ".mp4"
	
	convert_video_mp4(video_name, output_filename)
	delete_video(video_name)
	
	delete_first_name_from_file()	# delete first video name
	# loop and proceed next video in order
	
"""in_file_h264_before = "before.h264"
out_file_mp4_before = "before.mp4"

in_file_h264_after = "after.h264"
out_file_mp4_after = "after.mp4"


command1 = "ffmpeg -i " + video_folder + in_file_h264_before + " -vcodec libx264 -crf 28 " + video_folder + out_file_mp4_before
command2 = "ffmpeg -i " + video_folder + in_file_h264_after + " -vcodec libx264 -crf 28 " + video_folder + out_file_mp4_after

rm_command1 = "rm " + video_folder + in_file_h264_before
rm_command2 = "rm " + video_folder + in_file_h264_after

#"MP4Box -add " + file_h264 + " " + file_mp4
call([command1], shell=True)
call([rm_command1], shell=True)


print("\r\nRasp_Pi => Video Converted! \r\n")
call([command2], shell=True)
call([rm_command2], shell=True)
"""


"""
class Locker:
    def __enter__ (self):
        self.fp = open("./lockfile.lck")
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__ (self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
"""

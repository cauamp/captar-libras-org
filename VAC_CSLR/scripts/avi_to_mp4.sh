# Convert all .avi files in the current directory to .mp4 format using ffmpeg
find . -type f -name '*.avi' -exec ffmpeg -i {} -c:v libx264 -c:a aac -strict experimental "{}.mp4" \;
## created CMH 04-14 as a log to store useful commands


# function to replace all spaces in filenames with underscores
# (necessary for sorting with `ls -v`)

for f in *' '*
do
  mv -- "$f" "${f// /_}"
done



# to generate an animation of a given size, timing, and in sorted order
convert -resize 768x576 -delay 12 -loop 0 `ls -v` XX_animated.gif


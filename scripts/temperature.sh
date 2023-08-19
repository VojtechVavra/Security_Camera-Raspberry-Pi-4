#!/bin/bash
# -------------------------------------------------------
# Purpose: Display the ARM CPU and GPU  temperature of Raspberry Pi 2/3/4 
# -------------------------------------------------------
cpu=$(</sys/class/thermal/thermal_zone0/temp)
echo "$(date) @ $(hostname)"
echo "-------------------------------------------"
echo "GPU - $(/opt/vc/bin/vcgencmd measure_temp)"
# pokud cesta: neni nalezena, tak použít následující:
# /usr/bin/vcgencmd measure_temp
# program vcgencmd lze najít následovně: which vcgencmd
echo "CPU - $((cpu/1000))'C"

# ARM CPU temperature

# Old use
# measuring temperature, can be used without 'watch'
# watch vcgencmd measure_temp

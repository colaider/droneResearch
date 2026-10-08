# Open the drone's stereo camera from Windows with ROS 2 Jazzy installed (binary install, pixi).
#   .\view_stereo.ps1                                      # preview, ROS_DOMAIN_ID 42
#   .\view_stereo.ps1 -Domain 42 --fps 60 --exposure-us 3000 --gain 2
# Windows firewall must allow the Python/DDS UDP traffic (allow on the "Private" network prompt).
param(
    [string]$RosSetup = "C:\pixi_ws\ros2-windows\local_setup.ps1",
    [int]$Domain = 42,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)
. $RosSetup
$env:ROS_DOMAIN_ID = "$Domain"
python "$PSScriptRoot\..\rb3_stereo\rb3_stereo\viewer.py" --domain $Domain @Rest

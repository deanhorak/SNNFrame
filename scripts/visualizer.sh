#!/usr/bin/env bash
export DATA_PATH=~/remoterepo/SNNFrameData/recordings
export RECORDING=emnist_letters_full.snnr
export NETWORK=emnist_letters_full.snnw
export SWITCHES=$DATA_PATH/$RECORDING $DATA_PATH/$NETWORK --start-pause
cd ~/repos/SNNFrame
make -C build emnist_letters_visualized
cd ~/repos/SNNFrame/scripts
echo ./run_visualizer_playback.sh $DATA_PATH/$RECORDING $DATA_PATH/$NETWORK 
# --start-pause switch can be added to the run_visualizer_playback.sh script if desired 
./run_visualizer_playback.sh $SWITCHES




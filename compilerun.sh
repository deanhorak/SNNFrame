#!/bin/bash
cd "/home/dean/repos/SNNFrame"
cd /home/dean/repos/SNNFrame/build
make emnist_letters_training -j$(nproc) 2>&1 | tail -5 
cd experiments
rm -rf emnist_training_db
# LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 500 --test-limit 520 2>&1 | tail -50
# LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 500 --test-limit 520 2>&1 |& tee run_emnist.log
# LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --record /mnt/nas8t/SNNLogs/emnist_training.snnr --examples 4800 --test-limit 0 2>&1 |& tee run_emnist.log
#LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 4800 --test-limit 0 2>&1 |& tee run_emnist.log
#LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 5 --test-limit 260 2>&1 |& tee run_emnist.log
#LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 200 --test-limit 520 --seed 1 2>&1 |& tee run_emnist.log
 #LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu ./emnist_letters_training --examples 4800 --test-limit 0 2>&1 |& tee run_emnist_long.log
#LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu ./emnist_letters_training --examples 500 --test-limit 520 --seed 1 2>&1 |& tee run_emnist.log
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu ./emnist_letters_training --examples 500 --test-limit 520 --seed 1 --keep-l5-history 2>&1 |& tee run_test.log


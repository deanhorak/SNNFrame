 #!/usr/bin/env bash
cd /home/dean/repos/SNNFrame/build
rm -rf ./sonata_experiment_db_ab_B
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
timeout 7200 ./experiments/emnist_sonata_training \
  --config ../configs/emnist_v1_sonata/ab_valid100/circuit_config.json \
  --train-images ../data/EMNIST/emnist-letters-train-images-idx3-ubyte \
  --train-labels ../data/EMNIST/emnist-letters-train-labels-idx1-ubyte \
  --test-images ../data/EMNIST/emnist-letters-test-images-idx3-ubyte \
  --test-labels ../data/EMNIST/emnist-letters-test-labels-idx1-ubyte \
  --datastore ./sonata_experiment_db_ab_B \
  --max-passes 1 \
  --examples-per-class 200 \
  --test-limit 5200 \
  --seed 42 2>&1 | tee ab_similarity_B_valid100_final.log


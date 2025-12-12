#!/bin/bash
# Run all three learning strategy experiments

cd /home/dean/repos/snnfw

echo "=== Running Learning Strategy Experiments ==="
echo "Starting at: $(date)"
echo ""

# Create results directory
mkdir -p results/learning_strategies

# Run AppendStrategy
echo "Running AppendStrategy..."
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
SPDLOG_LEVEL=error \
./build/mnist_learning_strategies ./configs/mnist_learning_append.json \
2>&1 | tee results/learning_strategies/append_output.txt

# Run ReplaceWorstStrategy
echo ""
echo "Running ReplaceWorstStrategy..."
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
SPDLOG_LEVEL=error \
./build/mnist_learning_strategies ./configs/mnist_learning_replace_worst.json \
2>&1 | tee results/learning_strategies/replace_worst_output.txt

# Run MergeSimilarStrategy
echo ""
echo "Running MergeSimilarStrategy..."
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH \
SPDLOG_LEVEL=error \
./build/mnist_learning_strategies ./configs/mnist_learning_merge_similar.json \
2>&1 | tee results/learning_strategies/merge_similar_output.txt

echo ""
echo "=== All experiments complete ==="
echo "Finished at: $(date)"
echo ""
echo "Results summary:"
echo "----------------"
grep "Overall Accuracy" results/learning_strategies/*_output.txt


#ifndef SNNFW_EXPERIMENT_SUPERVISED_TEACHER_H
#define SNNFW_EXPERIMENT_SUPERVISED_TEACHER_H

#include "snnfw/Neuron.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/experiment/ExperimentConfig.h"
#include <memory>
#include <vector>

namespace snnfw {
namespace experiment {

/**
 * @brief Supervised teaching signal: forces the correct output population to spike.
 *
 * During training, fires the correct output neurons and learns patterns
 * only when STDP-derived eligibility criteria are met.
 */
class SupervisedTeacher {
public:
    explicit SupervisedTeacher(const ExperimentConfig& config);

    /**
     * @brief Apply supervised teaching for a given class label.
     *
     * Fires L5 winners, then forces the correct output population to spike
     * and learn the current pattern.
     *
     * @param classLabel The correct class label
     * @param columns Cortical columns (for L5 firing)
     * @param l5WinnerGlobal Which L5 neurons won
     * @param colHasL4 Which columns had L4 activity
     * @param outputPopulations All output populations
     * @param baseTime Current base time
     * @param propagator Network propagator
     * @return Number of patterns learned
     */
    int teach(int classLabel,
              std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
              const std::vector<bool>& l5WinnerGlobal,
              const std::vector<bool>& colHasL4,
              std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations,
              double baseTime,
              std::shared_ptr<NetworkPropagator> propagator);

private:
    const ExperimentConfig& config_;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_SUPERVISED_TEACHER_H
